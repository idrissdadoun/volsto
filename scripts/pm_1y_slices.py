"""The one-year addendum of the PM results package of 2026-10-09: the DJX expiries behind each
date's index target at the 12-month tenor — which the model's surface keeps, which the screen or
the calendar repair drops and why (``tables/1y_djx_slices.csv``, read by ``scripts/pm_1y.py``).

    NUMBA_NUM_THREADS=3 OMP_NUM_THREADS=1 python scripts/pm_1y_slices.py [--tenor 12m]
        [--base <folder>] [--dates <date> ...] [--timeout 120]

A **specification-only** build: for each distinct date of the addendum (the twenty yearly dates
of ``pm_1y.yearly_dates`` and the four reference dates) the inputs are loaded and the
specification is built as ``scripts/lcm_price.py`` builds it (``lcm_price.spec_for`` under
``configs/studies/dispersion/lcm.yaml``).  Nothing is calibrated and nothing is simulated.  The
SVI slice fits are read from the runs' fit records (``<cache>/svi_fits``) through a store that
never writes: a fit with no record is fitted in place and not stored.

One row per listed DJX expiry of the loader and date:

- ``status``: ``kept`` (a slice of the index surface as built), ``dropped`` (by the screen or
  by the calendar repair: ``rule`` and ``reason`` are the builder's record,
  ``info["dropped"]``) or ``not_selected`` (passes the screen and the repair, outside the
  surface's selection: shorter than ten days, or beyond the two slices after the horizon);
- ``svi_rms_vp``: the root-mean-square error of the slice's own SVI fit in vol points, for every
  expiry that passes the quote screen (``rms_source``: ``fit record`` or ``fitted here``); empty
  for an expiry the screen drops (it is never fitted);
- ``nearest_kept_below`` / ``nearest_kept_above``: the kept slice nearest the horizon on each
  side (a kept slice at the horizon itself counts as below);
- the date's own columns, repeated on each row: the horizon, the counts the row of
  ``lcm_price.py`` also carries (``n_dropped_index``, ``n_dropped_calendar_index``,
  ``index_last_slice``, ``index_extrapolated``, ``n_names_extrapolated``,
  ``n_names_unscreened``) with the names behind the two counts, and the specification key at
  each budget (``spec_key_development``, ``spec_key_production``: ``pm_1y.py`` compares them
  with the rows' ``spec_key``).

A date whose build raises or takes longer than ``--timeout`` seconds is written as one row with
``status = skipped`` and the reason.  The output folder follows ``pm_1y.output_base``:
``pm_update/1y`` until 13:00 New York time on 2026-10-09, ``pm_update/later/1y`` after.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

os.environ.setdefault("NUMBA_NUM_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcm_diagnostics as lcd
import lcm_price as lp
import pm_1y
import pm_common as pc

from volsto.calibration.cache import code_version
from volsto.calibration.fit_records import FitRecords
from volsto.calibration.lc_cache import lc_spec_key
from volsto.market.svi_slices import (
    FIT_MIN_POINTS,
    FIT_MIN_WIDTH,
    FIT_WIDTH_SD,
    MIN_SLICE_T,
    N_BEYOND,
    recorded_svi_fit,
    svi_fit_key,
    svi_fit_settings,
)
from volsto.studies.disp_lc import third_friday

log = logging.getLogger("pm_1y_slices")

TABLE = "1y_djx_slices"
BUDGETS = ("development", "production")


class ReadOnlyRecords(FitRecords):
    """The runs' fit records, read and never written: ``put`` returns the record it would have
    stored and leaves the store as it is."""

    def put(
        self,
        key: str,
        inputs: Mapping[str, Any],
        summary: Mapping[str, Any],
        *,
        origin: str,
        code_tag: str = "",
    ) -> dict[str, Any]:
        stored = self.get(key)
        return stored if stored is not None else {"key": key, "fit": dict(summary)}


class BuildTimeoutError(Exception):
    pass


def _alarm(signum: int, frame: FrameType | None) -> None:
    raise BuildTimeoutError


def slices_of(date: str, tenor: str, cfg: dict[str, Any], records: FitRecords) -> pd.DataFrame:
    """The DJX expiries of ``date`` (module docstring), one row each."""
    t0 = time.perf_counter()
    inp = lcd.load_inputs(date, tenor, cfg["index"])
    keys: dict[str, str] = {}
    info: dict[str, Any] = {}
    for budget in BUDGETS:
        spec, info = lp.spec_for(inp, cfg, budget, records)
        keys[budget] = lc_spec_key(spec)
    horizon = float(inp.T)
    kept_times = [float(t) for t in info["index_slices"]]
    gone = {g["expiry"]: g for g in info["dropped"] if g["leg"] == "index"}
    listed = {e.expiry for e in inp.index_smiles}
    settings = svi_fit_settings(FIT_WIDTH_SD, FIT_MIN_WIDTH, FIT_MIN_POINTS)
    below = [t for t in kept_times if t <= horizon + 1e-9]
    above = [t for t in kept_times if t > horizon + 1e-9]
    rows = []
    for e in sorted(inp.index_smiles, key=lambda x: float(x.T)):
        t = float(e.T)
        drop = gone.get(e.expiry)
        is_kept = drop is None and any(abs(t - k) < 1e-12 for k in kept_times)
        rms, source = None, ""
        if drop is None or drop["rule"] == "calendar":
            # the slice's own fit, as the builder's (one record per slice: the same number
            # whatever the other slices are)
            hit = records.get(svi_fit_key(t, e.k, e.vol, settings)) is not None
            fit = recorded_svi_fit(e.k, e.vol, t, records=records, origin="pm_1y_slices:index")
            rms, source = float(fit.rms_vp), "fit record" if hit else "fitted here"
        rows.append(
            {
                "expiry": e.expiry,
                "T": t,
                "third_friday": bool(third_friday(e.expiry, listed)),
                "status": "kept" if is_kept else ("dropped" if drop else "not_selected"),
                "rule": drop["rule"] if drop else "",
                "reason": drop["reason"] if drop else "",
                "svi_rms_vp": rms,
                "rms_source": source,
                "side_of_horizon": "above" if t > horizon + 1e-9 else "below",
                "nearest_kept_below": bool(is_kept and below and abs(t - max(below)) < 1e-12),
                "nearest_kept_above": bool(is_kept and above and abs(t - min(above)) < 1e-12),
            }
        )
    frame = pd.DataFrame(rows)
    frame.insert(0, "date", date)
    frame.insert(1, "tenor", tenor)
    frame.insert(2, "horizon", horizon)
    seconds = time.perf_counter() - t0
    for key, value in {
        "n_index_listed": len(inp.index_smiles),
        "n_index_kept": len(kept_times),
        "n_dropped_index": len(gone),
        "n_dropped_calendar_index": int(info.get("n_dropped_calendar_index", 0)),
        "index_last_slice": max(kept_times),
        "index_extrapolated": bool(info["index_extrapolated"]),
        "n_names_extrapolated": int(info["n_names_extrapolated"]),
        "names_extrapolated": ",".join(info["names_extrapolated"]),
        "n_names_unscreened": int(info["n_names_unscreened"]),
        "names_unscreened": ",".join(info["names_unscreened"]),
        "spec_key_development": keys["development"],
        "spec_key_production": keys["production"],
        "selection": (
            f"expiries of at least {MIN_SLICE_T * 365:.0f} days up to the horizon "
            f"and the next {N_BEYOND}"
        ),
        "build_seconds": round(seconds, 2),
    }.items():
        frame[key] = value
    return frame


def skipped_row(date: str, tenor: str, reason: str) -> pd.DataFrame:
    """The one row of a date the build could not do."""
    return pd.DataFrame([{"date": date, "tenor": tenor, "status": "skipped", "reason": reason}])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="12m")
    ap.add_argument("--base", default=None, help="the output folder (default: pm_1y's)")
    ap.add_argument("--dates", nargs="*", default=None, help="default: the addendum's dates")
    ap.add_argument("--config", default=str(lp.CONFIG))
    ap.add_argument("--timeout", type=int, default=120, help="seconds per date")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for name in ("volsto.studies.disp_lc", "lcm_diagnostics", "lcm_price"):
        logging.getLogger(name).setLevel(logging.WARNING)
    now = dt.datetime.now(pm_1y.NY)
    base, _ = pm_1y.output_base(args.base, now, test=args.tenor != "12m")
    cfg = lp.load_config(args.config)
    # the runs' fit records (lcm_price.run_date: <root>/<cache>/svi_fits), read only
    records = ReadOnlyRecords(pc.LC_OUT.parents[1] / cfg["cache"] / "svi_fits")
    if args.dates:
        dates = list(args.dates)
    else:
        yearly = pm_1y.yearly_dates(pm_1y.study_entries(args.tenor), args.tenor)
        dates = sorted({*(d for d in yearly["date"] if d), *pc.REFERENCE_DATES})
    commit = code_version()
    built = now.strftime("%Y-%m-%d %H:%M")
    frames = []
    signal.signal(signal.SIGALRM, _alarm)
    for date in dates:
        t0 = time.perf_counter()
        signal.alarm(args.timeout)
        try:
            frame = slices_of(date, args.tenor, cfg, records)
        except BuildTimeoutError:
            frame = skipped_row(
                date, args.tenor, f"the specification build took more than {args.timeout} s"
            )
        except Exception as exc:
            frame = skipped_row(
                date, args.tenor, f"the specification build raised {type(exc).__name__}: {exc}"
            )
        finally:
            signal.alarm(0)
        n = frame["status"].value_counts().to_dict()
        log.info("%s: %s in %.1f s", date, n, time.perf_counter() - t0)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["config"] = str(Path(args.config).resolve())
    out["git_commit"] = commit
    out["built"] = built
    path = pc.save_table(out, TABLE, base=base)
    skipped = sorted(out[out["status"] == "skipped"]["date"])
    names = f": {', '.join(skipped)}" if skipped else ""
    log.info("%d dates (%d skipped%s) -> %s", len(dates), len(skipped), names, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
