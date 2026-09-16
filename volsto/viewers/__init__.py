"""Viewers layer (SPEC §9, M9): the precompute CLI, the results store, the read API and the
Streamlit pages.

Repository convention: the owner's ``viewers/...`` paths are this package, ``volsto/viewers/``
(library code), with the console entry points in ``pyproject.toml`` ``[project.scripts]``
(``volsto-precompute`` → :func:`volsto.viewers.precompute.main`).

Standing rules of the layer: the precompute CLI is the one place that calibrates a leverage
(through :meth:`~volsto.calibration.cache.LeverageCache.get_or_calibrate`); the store, the API
and the pages only read the results store and the leverage cache, and a missing point is
reported with the exact ``volsto-precompute ...`` command that produces it.  Every Monte Carlo
number the store exposes carries its standard error (a ``<name>_stderr`` column).
"""
