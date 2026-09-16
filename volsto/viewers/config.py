"""Viewer configuration (SPEC §9.2 "Read API and app", M9 Part 3): where the read-only layer
finds the leverage cache, the results store, the M7 / M8b output files and the surface
snapshots — and nothing else, because the pages and the API never compute.

:class:`ViewerConfig` is resolved from four sources, later ones winning:

1. the documented defaults below (repository-root-relative, so the viewer works from any cwd);
2. the optional YAML ``configs/viewer.yaml`` (:data:`DEFAULT_CONFIG_FILE`, or the file named by
   ``VOLSTO_VIEWER_CONFIG``) with the keys ``cache_root, store_root, outputs_root,
   snapshots_root, grid_path`` — unknown keys are a :class:`~volsto.config.ConfigError`;
3. the environment variables :data:`ENV_VARS` (``VOLSTO_CACHE``, ``VOLSTO_STORE``,
   ``VOLSTO_OUTPUTS``, ``VOLSTO_SNAPSHOTS``, ``VOLSTO_GRID``);
4. the CLI flags of ``volsto-viewer`` (``--cache``, ``--store``, ``--outputs``,
   ``--snapshots``, ``--grid``), which :mod:`volsto.viewers.app` forwards to the Streamlit process
   through the same environment variables, so a page always reads
   :meth:`ViewerConfig.from_env`.

A relative path given by the user is kept relative to the current working directory (the
``volsto-precompute`` convention: ``--store outputs/store --cache cache``); the defaults are
absolute under the repository root.  ``grid_path`` is the grid YAML the pages name in the
``volsto-precompute`` command of a missing point.

Checked by ``tests/test_viewers_api.py`` (``test_config_precedence``).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from volsto.config import ConfigError
from volsto.viewers.grid import REPO_ROOT

#: Leverage cache root (``LeverageCache``), the repository cache by default.
DEFAULT_CACHE_ROOT = REPO_ROOT / "cache"
#: Results store root (``ResultsStore``): the ``volsto-precompute`` default ``outputs/store``.
DEFAULT_STORE_ROOT = REPO_ROOT / "outputs" / "store"
#: Root of the study outputs the marking (``m7/``) and hedging (``m8b/``) pages browse
#: (git-ignored; the pages name the script that produces a missing file).
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"
#: Surface snapshots (``market`` / ``ssvi`` YAML documents).
DEFAULT_SNAPSHOTS_ROOT = REPO_ROOT / "configs" / "surfaces" / "snapshots"
#: The grid the pages refer to (surface catalogue and the precompute command).
DEFAULT_GRID_PATH = REPO_ROOT / "configs" / "grids" / "default.yaml"
#: Optional YAML overriding the defaults.
DEFAULT_CONFIG_FILE = REPO_ROOT / "configs" / "viewer.yaml"
#: Environment variables, field → name.
ENV_VARS: dict[str, str] = {
    "cache_root": "VOLSTO_CACHE",
    "store_root": "VOLSTO_STORE",
    "outputs_root": "VOLSTO_OUTPUTS",
    "snapshots_root": "VOLSTO_SNAPSHOTS",
    "grid_path": "VOLSTO_GRID",
}
ENV_CONFIG_FILE = "VOLSTO_VIEWER_CONFIG"
#: CLI flag per field (``volsto-viewer --cache ...``).
CLI_FLAGS: dict[str, str] = {
    "cache_root": "--cache",
    "store_root": "--store",
    "outputs_root": "--outputs",
    "snapshots_root": "--snapshots",
    "grid_path": "--grid",
}


@dataclass(frozen=True)
class ViewerConfig:
    """Roots of the read-only layer (module docstring)."""

    cache_root: Path = DEFAULT_CACHE_ROOT
    store_root: Path = DEFAULT_STORE_ROOT
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT
    snapshots_root: Path = DEFAULT_SNAPSHOTS_ROOT
    grid_path: Path = DEFAULT_GRID_PATH

    # -- construction ------------------------------------------------------------------------

    @classmethod
    def from_sources(
        cls,
        *,
        yaml_path: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        overrides: Mapping[str, str | Path | None] | None = None,
    ) -> ViewerConfig:
        """Defaults ← YAML (``yaml_path`` when it exists) ← ``env`` ← ``overrides`` (CLI)."""
        values: dict[str, Any] = {}
        path = Path(yaml_path) if yaml_path is not None else None
        if path is not None and path.exists():
            values.update(_read_yaml(path))
        env = os.environ if env is None else env
        for field_name, var in ENV_VARS.items():
            if env.get(var):
                values[field_name] = env[var]
        for field_name, v in (overrides or {}).items():
            if field_name not in ENV_VARS:
                raise ConfigError(f"unknown viewer setting {field_name!r}")
            if v is not None:
                values[field_name] = v
        return cls(**{k: Path(v) for k, v in values.items()})

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ViewerConfig:
        """The configuration a page reads: YAML (``VOLSTO_VIEWER_CONFIG`` or the default file)
        then the environment."""
        env = os.environ if env is None else env
        yaml_path = env.get(ENV_CONFIG_FILE) or DEFAULT_CONFIG_FILE
        return cls.from_sources(yaml_path=yaml_path, env=env)

    @classmethod
    def from_args(
        cls, argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None
    ) -> tuple[ViewerConfig, argparse.Namespace]:
        """Parse the ``volsto-viewer`` flags on top of :meth:`from_env`; returns the config and
        the remaining namespace (``port``, ``headless``)."""
        args = build_parser().parse_args(argv)
        env = os.environ if env is None else env
        yaml_path = args.config or env.get(ENV_CONFIG_FILE) or DEFAULT_CONFIG_FILE
        overrides = {k: getattr(args, k) for k in ENV_VARS}
        return cls.from_sources(yaml_path=yaml_path, env=env, overrides=overrides), args

    # -- export ------------------------------------------------------------------------------

    def to_env(self) -> dict[str, str]:
        """The environment variables reproducing this configuration."""
        return {var: str(getattr(self, field_name)) for field_name, var in ENV_VARS.items()}

    def as_dict(self) -> dict[str, str]:
        return {f.name: str(getattr(self, f.name)) for f in dataclasses.fields(self)}

    def relative_to_cwd(self, p: Path) -> str:
        """``p`` printed relative to the cwd when it lies below it (shorter commands)."""
        try:
            return str(p.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(p)


def _read_yaml(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the viewer config must be a mapping")
    unknown = sorted(set(data) - set(ENV_VARS))
    if unknown:
        raise ConfigError(f"{path}: unknown viewer settings {unknown}; known: {sorted(ENV_VARS)}")
    return {str(k): str(v) for k, v in data.items()}


def build_parser(prog: str = "volsto-viewer") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Streamlit viewer over the results store and the leverage cache (read-only; a "
            "missing point prints the volsto-precompute command that produces it)."
        ),
    )
    for field_name, flag in CLI_FLAGS.items():
        p.add_argument(
            flag,
            dest=field_name,
            default=None,
            help=f"{field_name} (env {ENV_VARS[field_name]}; default "
            f"{getattr(ViewerConfig(), field_name)})",
        )
    p.add_argument("--config", default=None, help=f"viewer YAML (env {ENV_CONFIG_FILE})")
    p.add_argument("--port", type=int, default=8501, help="Streamlit server port")
    p.add_argument("--headless", action="store_true", help="do not open a browser")
    return p
