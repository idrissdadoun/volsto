"""Rolling date-by-date backtest (SPEC §10.3, M10 Part 3; console script ``volsto-backtest``).

**Two stages, one separation of concerns.**

1. ``volsto-backtest run <config>`` is the **only M10 path that calibrates leverage**.  For each
   trading date of the configured range it imports and fits the surface (eSSVI, calendar repair
   on), runs the P1 marking fit, calibrates the date's leverage through the cache (a hit costs
   nothing), prices the book, runs the P&L attribution from the previous date, and stores the
   date's outcome — one row per (date, trade), the fit record and ``done.json`` — as an
   immutable attempt (*Storage* below).  Shardable (``--shard i/n``), resumable (``--resume``),
   and it prints the projected wall clock before any work.  ``volsto-backtest status``,
   ``migrate`` and ``gc`` inspect, migrate and clean the store.
2. This module is also a **study module** of :mod:`volsto.studies.runner` (``volsto-study run
   configs/studies/catalogue/backtest_2022h2.yaml``): it reads the per-date store and never
   calibrates (the runner's guard and cache observation hold); every table and figure is drawn
   from ``results.parquet``.  An unconfirmed date is a requirement of kind ``artefact`` (its
   ``CURRENT`` pointer) whose command is the exact ``volsto-backtest run <config> --only-dates
   ...`` line (``migrate`` for a store of the flat layout).  **Sign:** the stored rows keep the
   holder's sign, ``V(d) − V(d−1) + flows``; every table, figure and sentence of the study
   presents the **desk** P&L, minus that (the desk is short the book, the M7 / M8b convention;
   :data:`DESK_SIGN`).

**Config** (:class:`BacktestConfig`, strict YAML — :class:`volsto.studies.runner.StrictLoader`;
every key is required, an unknown key is an error; ``configs/backtest/hdn_2022h2.yaml`` is the
annotated reference)::

    name: hdn_2022h2
    data: {vendor: hdn, root: data/hdn_sample/options_sample_2022H2, underlying: SPX,
           missing_close: fail | skip_date}
    dates: {start: "2022-07-01", end: "2022-12-30"}
    surface: {essvi: true, calendar_repair: true}
    marking: {ssr_target: 1.0, skew_eps: 0.10, stage3: false}
    calibration: {base_spec: configs/studies/lsv_reference_2f.yaml, n_particles: 800000,
                  horizon: 3.0}
    pricing: {n_paths: 20000, chunk_size: 20000, seed: 2024}      # one seed: CRN across dates
    book:
      fixed: [{id, kind, maturity, ...}, ...]          # struck on the first date of the range
      rolling: {every: month | none, mark: daily | inception, products: [...]}
    attribution: {mode: sticky_leverage, detail: ladders, trades: fixed | all | none,
                  pillars: [...], bump: 0.01, max_halvings: 6}
    ssr: {window: 20, pillars: [0.25, 0.5, 1.0]}
    stability: {window_vol: 60, window_ssr: 60, share: 0.5, band: 1.0,
                fit: {pillars, mat_min, k2, skew_mode, skew_eps, nu_cap, ...}}
    paths: {out: ..., cache: cache, snapshots: ...}

Trade kinds (:data:`TRADE_KEYS`): ``autocall`` / ``phoenix`` (the M6 headline term sheets of
:mod:`volsto.studies.m6` with ``observations`` equally spaced dates to ``maturity``; at 3y / 3
they are :func:`~volsto.studies.m6.headline_products` exactly), ``cliquet`` (the study cliquet,
monthly, cap 2%, global floor 0), ``vko_put`` (strike ``moneyness`` × spot, ``vol_ko``, daily
fixings, notional 1/spot), ``ko_var`` (up barrier ``barrier`` × spot) and ``var_swap``, both with
``strike: vs_strip`` (the inception surface's log-contract strike at the maturity) or a strike
vol, variance notional ``1 / (2 K_vol)`` (values in vol units of vega notional 1; M8b's KO var
had strike 0 in variance units instead); ``ko_var`` takes the optional ``settlement``
(``maturity`` by default, ``knock_out``: paid at the knock-out close, the desk's convention).
The payoff study's kinds (2026-09-27): ``uo_call`` / ``do_put`` / ``di_put`` (strike
``moneyness`` × spot, barrier ``barrier`` × spot, daily-close monitoring, strict, notional
1/spot: % of the inception spot), ``var_put`` (put on realised variance ``(K² − RV)⁺`` struck at
``strike_ratio`` times the ``strike`` rule's vol, variance notional ``1 / (2 K_vol)``),
``up_var`` / ``down_var`` (corridor variance swaps above / below ``barrier`` × spot, the desk
indicators ``prev`` / ``curr``, struck by the ``strike`` rule, variance notional
``1 / (2 K_vol)``).  Coupons are per annum: the autocall pays ``c · T_i``
at observation ``i`` and the Phoenix ``c · (T_i − T_{i−1})`` per period, which are the M6 float
coupons when the observations are annual.  Every fixing
lies on the 252-day trading grid of :mod:`volsto.products.seasoning` (checked at load).  Values
are stored in product units; the ``scale`` / ``unit`` columns convert (× 100: % of notional, % of
the inception spot, vol points of vega notional).

**The config hash** (:meth:`BacktestConfig.content_hash`) is the SHA-256 of the canonical JSON
of the mapping without ``paths``, ``data.root`` and ``dates.end`` (relocating the store, the
cache or the data, and extending the range, leave computed dates valid), with the **content** of
the base spec (particle, simulation, local-vol and perturbation settings — not its path) and the
resolved marking and stability fit configurations.  ``run`` refuses a store whose
``backtest.json`` hash differs (exit 2) unless ``--force``.

**Storage** (:class:`BacktestStore`).  Per date, ``dates/<date>/attempts/<id>/`` holds each
outcome ever written — ``done.json`` (status, dependency record, bookkeeping, and the writing
code's version, informational only) and, for results and skips, ``rows.parquet`` and
``fit.json`` — complete and immutable: it is staged in ``attempts/.staging-<host>-<pid>-*`` and
published by ONE rename, its id ``<status>-<digest>`` the SHA-256 of its file listing (any
attempt verifies on its own).  The one mutable object is ``dates/<date>/CURRENT``, a pointer
(version, attempt id, status, the SHA-256 of every file) replaced by ONE ``os.replace`` of a
synced temporary, the directory synced; publishing and pointing happen under the date's lock:
``flock`` on the date DIRECTORY, every mutation made through that locked descriptor
(``dir_fd``-relative renames, replaces, unlinks and removals; :meth:`BacktestStore.lock` argues
why a swapped path cannot split the lock).  A ``CURRENT`` of another pointer version (a newer
volsto-backtest) is a refusal for every mutating command, never damage to overwrite.  Attempt
checks ignore exactly Finder's ``.DS_Store`` and the AppleDouble ``._<name>`` of a file the
attempt holds — regular files only (a directory of any name is listed and fails the check).
**The pointer rule** (:func:`choose_pointer`, one place): attempts are ranked — done > a pending
ok > incomplete > a pending incomplete > any other attempt whose files verify and hold results
(stale ones included) > anything without results; ``CURRENT`` moves to the new attempt when it
ranks at least as high as the current one and as any attempt of this config, else to the best
attempt of this config when that ranks higher; an attempt without results never becomes current
while any attempt of the date holds results, and an attempt of another config is never adopted.
The decision is taken under the date's lock from a fresh read.  Nothing moves while a verdict
involved is ``unsettled``.  A failed or skipped attempt next to results is only recorded
(``status`` notes a later failure, read from verified bytes only and never part of a
classification; a skip still drops its date from the calendar when its own record verifies).
``run`` never deletes an attempt; after its refusals and after writing its skips it **adopts**
(:func:`adopt`) for each selected date the attempt the rule prefers — what a writer killed
between publishing and pointing left.  ``volsto-backtest gc`` (:func:`collect_garbage`) is
conservative and bound to the config: refused (before any write) unless the store header names
the given config; under every date's lock it applies the rule, then removes the non-current
attempts only of dates whose full verdict is done (each holds no results or is superseded by the
current done attempt; symbolic links skipped), plus pointer temporaries, the staging directories
of dead writers on this host (other hosts' are kept and counted: ``status`` lists them) and
migrated leftovers.  A store of the flat layout of earlier versions
(``dates/<date>/{rows.parquet, fit.json, done.json, failure.json}`` and ``<out>/.staging``) is
migrated in place by ``run`` (after all its refusals, evaluated on the flat files), ``status``,
``migrate`` or ``gc`` — refused (exit 2, nothing written) unless its header names the given
config (:func:`migrate_store`: under the store lock, JOURNALLED in ``migrations.json`` — an
``in_progress`` entry before the first destructive step, each date's report after it, the
binding and the final record resumable; files copied into ``attempts/legacy-<status>-<digest>``;
``CURRENT`` set by the pointer rule alone; the flat files removed last and those no attempt
holds moved to ``quarantine/``; leftovers renamed ``*.imported`` before removal; snapshots bound
only when a fresh import reproduces them); a flat store that cannot be written is a clear error
before any lock; stage 2 asks for the migration.  **The flat-layout code (volsto-backtest before
the attempts) and this code must never write the same store** — the date lock does not survive
the directory renames of the flat layout.  Every step of the commit and migration paths calls
:data:`CRASH_HOOK`, and a lock-free reader calls :data:`READ_HOOK` after each read (tests inject
crashes and races there).

**Integrity** (:class:`Ledger`, :func:`dependency_record`).  A date is ``done`` iff its
``CURRENT`` pointer verifies — the reader reads ``CURRENT`` and every file of the attempt it
names ONCE, checks those bytes against the recorded hashes and parses the same bytes (never the
files again; an attempt that vanishes while ``CURRENT`` changes is read again), so no byte of an
outcome (rows, fit, any field of ``done.json``) can change unnoticed — and ONE function,
:func:`dependency_record`, recomputes the attempt's complete dependency record from the world as
it is and the result equals the record stored in its ``done.json``: the config hash; the inputs
digest (:class:`InputIndex`: the chain over each calendar date's day-file checksum and manifest
entry up to the date); the calibration code tag; the digest of the date's snapshot
(:func:`snapshot_digest`: the file it was marked from, its creation-time line aside); the
checksum and row count of ``rows.parquet`` and the checksum of ``fit.json``; the leverage key
recomputed from the snapshot, the stored fit and the base spec, complete in the configured
cache, and the digest of its NUMBERS (:func:`leverage_content_digest`: every array of
``leverage.npz`` but its metadata, memoised by file identity; record version 2 — version-1
records of earlier stores are still judged at their version); the previous calendar date and
whether its leverage was used; the **state link** (:func:`state_link`: snapshot digest, fitted
parameters, leverage key and, version 2, leverage content) of every date whose marked state the
rows use — the previous date and the live trades' inception dates — and the snapshot digest of
every date whose close is in a live trade's realised history (:func:`dependencies_of`); for a
skip, its cause (the day file is absent), re-derived.  ``run`` records what it USED
(:class:`_UsedInputs`: the digests of the snapshot bytes it parsed, the content digest of the
leverage bytes it priced with, read once by :class:`_RecordingCache`); nothing is read again at
commit.  The verdicts chain: a date is confirmed only when its previous date and its state
dependencies are, so a changed input makes its date and every later date stale, and recomputing
a date into another state (another spot, another fit) makes every date that used its state or
its close stale.  The chain carries states, not files — the rows and the fit of a date carry
timings, so a bit-identical recomputation of d−1 leaves d confirmed; it is the only chain
contiguous shards can satisfy, since a block's first date is computed from its predecessor's
state before the other shard stores that predecessor.  Until then such a date is ``pending``
(``--resume`` leaves an ``ok`` one alone; stage 2 requires the date it waits for). ``status``,
``run --resume`` (its selection and a check before each date, so a date confirmed by the dates
recomputed before it is skipped), both refusals of ``run``, the pointer rule and the study's
requirements and selection all read :meth:`Ledger.verdict`.  Statuses: ``done``; ``incomplete``
(valuations stored, some P&L missing — the previous date unavailable, e.g. its leverage missing
under ``--no-calibrate``, or rows unpriced; the reasons and the recompute command are stored;
``--resume`` recomputes it; exit 2 when a missing leverage is the cause, else 1); ``failed``;
``skipped`` (below); ``pending``; ``unsettled`` (a consistent read could not be had while
writers changed the store: never acted on, never stale); ``stale`` (the first mismatch is the
reason); ``missing``.  **Snapshots are bound to the vendor**: at import, an import record
(``<snapshots>/<underlying>_<date>.import.json``, :func:`bind_snapshot`) stores the snapshot's
content digest with the day-file and manifest-entry digests; a snapshot is reused only when both
still match (:func:`snapshot_bound`) — an edited snapshot is re-imported.

**Refusals** (exit 2, before ANY write — header, migration, skip, adoption, probe, import;
each prints the commands that would proceed).  A ``CURRENT`` of another pointer version
(:func:`foreign_pointer_refusal`).  A store whose ``backtest.json`` hash differs
(:func:`header_refusal`), and a date to compute or its previous date whose stored outcome — in
attempts or still in the flat layout — was computed under another config hash, unless
``--force`` (:func:`hash_refusals`, through :func:`computed_under_another_config`, the one
judgement stage 2's commands use too).  Under
``--no-calibrate`` (:func:`leverage_refusals`): a date to compute whose current results name a
leverage the configured cache lacks; the previous date of a date to compute — with or without
stored results of its own — when its leverage is lacking and it has results or the date's stored
P&L used it (its key from the date's ``previous_cache_key``, or by marking it).

**Missing closes** (``data.missing_close``, required).  The calendar is the vendor's day files
inside ``[start, end]`` plus the dates its manifest lists as ``trading_days_missing``.  ``run``
imports every date up to its last one first (the realised histories need every close).  A date
whose close cannot be had fails with a message naming the date and the fix, a trade whose
history spans it is ``unpriced`` (its dates are ``incomplete``), and the trades struck after it
proceed — under ``skip_date`` too when the day file exists (an import that raises is a failure,
retried by ``--resume``, never a calendar gap).  With ``skip_date`` a date whose day file is
ABSENT (the one deterministic cause, :data:`SKIP_CAUSE`, recorded and re-derived by the verdict)
is dropped from the calendar before the shard blocks are cut (a pure check of the inputs;
``done.json`` status ``skipped``): the 252-day fixing grid then runs over the remaining vendor
dates, so every later fixing of a trade spanning the gap lands one exchange day late and the
return across the gap counts as one daily return — the rows record their ``gaps`` and study.md
states it.  A day that is neither present nor listed by the manifest is invisible (no exchange
calendar is available).

**Dates and sharding.**  The k-th calendar date after a trade's inception is its trading index
k (the seasoning date map).  The fixed
book is struck on the first calendar date, the rolling book on the first calendar date of each
month.  ``--shard i/n`` takes the i-th of n **contiguous** blocks (``numpy.array_split``) of the
``--only-dates`` selection (``YYYY-MM-DD`` or ``A..B`` ranges; the calendar without it) — not
the interleaved shards of ``volsto-precompute``: date d's attribution needs date d−1's
leverage, so interleaved shards run in parallel would calibrate each other's dates; with blocks
only a block's first date needs its predecessor, one extra (duplicate) calibration per boundary.
The union of the shards is the unsharded selection.  A date job needs the closes of every
earlier date (the realised history), so ``run`` first imports every missing snapshot up to its
last date (``--snapshots``; about 1 s per day; the whole selection under ``skip_date``).

**Per-date job** (:meth:`BacktestRun.run_date`).  (1) The snapshot (import once, reused after;
re-imported when its source checksum, eSSVI flag or repair provenance disagree); the spec is
``snapshot_spec(base_spec, snapshot)`` with the configured particles and horizon, the surface is
rebuilt from the snapshot (so a date's fit is a function of the file alone, whichever process
computes it).  (2) :func:`~volsto.calibration.fit_2f.fit_2f_marking` at ``(ssr_target,
skew_eps)``, stage 3 **off** (the SPX (1.0, 0.10) fit fails the stage-3 engine-bias assertion,
SPEC §15; the fit's first-order SSR is what is marked); an ``infeasible`` fit fails the date (no
model, as ``volsto-precompute`` does).  The per-pillar history quantities (VS vol, ATMF vol,
ATMF skew, ln spot) are stored for the realised SSR.  (3) The leverage through
``LeverageCache.get_or_calibrate`` (``--no-calibrate``: cache only, under
:func:`volsto.calibration.guard.calibration_forbidden`; a miss fails the date with the command).
(4) Each live trade is seasoned to the date (:func:`volsto.products.seasoning.replay`, the
date's discount curve) and priced on one engine whose builder is based on the previous date's
state (so the previous date's value, the attribution and today's value share one memo and one
seed); a settled trade carries its deterministic cash.  (5) The P&L of ``(d−1, d]`` is ``V(d) −
V(d−1) + flows dated d``: for an attributed trade by :func:`volsto.risk.attribution.explain` in
the configured mode and detail (``dt = 1/252``, ``product_1`` the trade seasoned to d,
``product_theta`` the trade seasoned with d's close held at d−1's), whose buckets plus the cash
flows sum to it exactly; for the other trades by the paired CRN difference; on a settlement day
by the settled cash (bucket ``settlement``).  The start-of-period Greeks the attribution priced
(sticky-moneyness delta and gamma, the frozen-leverage vega and ladders, rho, repo, theta and
its split, the parameter sensitivities of the parameters that moved) are read back from the
engine's memo at no pricing cost and stored with their standard errors, as is every bucket's
standard error (the paired one of the per-path sum of the bucket's items,
:attr:`volsto.risk.attribution.Explain.bucket_stderrs` — a ladder bucket's pillars share the
seed; a single item's is |move| × the Greek's; the rates buckets are directional sensitivities
along the day's curve moves, :func:`volsto.risk.attribution.curve_move_sensitivities`), with the
paired errors of the bucket groups (``grp_<group>_stderr``, :data:`PAIRED_GROUPS`) and a flag
when the residual's error had to be a root sum of squares.  The P&L's standard error is the
direct paired one of ``V(d) − V(d−1)`` (both prices are in the engine's memo) and the residual's
is paired per path as well (:attr:`volsto.risk.attribution.Explain.residual_stderr`): a root sum
of squares of the steps' errors ignores their covariance (0.72–1.79× the paired error on the
2022-10-27..11-01 rows). For every attributed trade the row also stores the cumulative P&L
``V(d) − V(inception) + flows`` with its DIRECT paired error (``cum_pnl``, ``cum_pnl_stderr``):
the date's value and the fresh trade at the inception state priced on common random numbers, one
extra pricing per trade and date (none the day after inception, a memo hit), the inception state
and leverage from the cache (calibrated if missing, except under ``--no-calibrate``: the date is
then incomplete). (6) The rows (:data:`ROW_COLUMNS`), the fit record and ``done.json`` (status,
hash, digest, checksums, key, wall clock per step, whether this process calibrated).

**Projection** (:func:`project`, printed by ``dry-run`` and before ``run`` works).  A probe on
the first two consecutive calendar dates that can be marked (imported into a temporary directory
unless their snapshots exist) measures:

* the import and fit seconds;
* one model build with its ξ₀ strip (``build_market`` with the memo off, about 1.7 s) and one on
  a memo hit (``volsto.calibration.cache.XI0_MEMO``: a few ms);
* the best of two timings of one pricing of every book product at ``p = min(n_paths, 20000)``
  and at ``p/5`` paths, on a flat-leverage LSV of the probe date (the pricing cost does not depend
  on the leverage values); seconds per pricing are the affine fit through the two sizes,
  extrapolated to ``n_paths``.

A dry run of :func:`~volsto.risk.attribution.explain` on the probe pair
(:class:`~volsto.risk.attribution.DryRunEngine`) counts each attributed trade's pricings, the
distinct states of a date and their distinct ξ₀ strips (parameter bumps share their state's
strip; the previous date's was stripped the day before).  A date's builds cost ``states ×
warm build + new strips × strip``.  A calibration is charged to every date whose key is not
known to be cached (:func:`scaled_calibration_s`: the cache manifest's median at the configured
particles and horizon, else the nearest size or the 140 s / 8·10⁵ / 3y reference scaled by
particles × steps, never below :data:`CALIBRATION_MIN_S`; none under ``--no-calibrate``).
The cumulative P&L is charged one pricing per attributed trade per date, and a block start the
fit and calibration of every earlier inception of an attributed trade live there.
Shards: the slowest contiguous block of the selection (+ its boundary calibration) and the sum,
for 1, 2, 4, 8 and the requested count, each process at the probe's thread count.  ``run``
reuses ``<out>/probe.json`` for the same config hash and thread count.

**Study (stage 2).**  ``params: {backtest: <config>, store: <store dir relative to the outputs
root, or null>}``; the study's ``seeds.pricing`` must equal the backtest's.  A stale date is a
requirement with its command (exit 2): a flat store is migrated first, and every date to
recompute goes into one ``run`` line carrying ``--force`` exactly when ``run`` would refuse it
otherwise; an ``unsettled`` date asks for a rerun once the writers finish.  The study pins the
``CURRENT`` bytes it read.  Results: the window as the data shows it (dates, first
and last close, return, realised vol; no regime adjective); the dates whose P&L is incomplete,
with their reasons and commands (never dropped); the book and inception prices (one row per
trade id, each in its own unit: % notional, % of the inception spot, vol points); the
cumulative P&L per trade and per bucket, and per month, with a book total only over the fixed
trades quoted in one unit (a total never adds units; tables and figures split by unit); the
daily series (figures); the fitted parameters with
:func:`~volsto.calibration.stability.flag_unidentified` on the marking fits — whose standard
errors are NaN by construction (the marking targets carry no sampling error), so the flags there
report "no usable standard error" — and the §15 Part 4 historical-mode rolling fit
(:func:`~volsto.calibration.stability.rolling_fit`, windows and ``stability.fit`` from the
config, stated in study.md) with its flags once
the history is long enough; the realised SSR (rolling window from the stored pillar history, NaN
with the reason before ``window + 1`` dates) against the marked target and the fit's first-order
SSR; the VKO's marked price against its realised state (the realised outcome of a 12m trade is
not observable inside the 2022 H2 window: stated); the fit statuses and the cost.  The
monthly table carries every bucket and a ``check`` column (total minus their sum).  Every
aggregate goes through :func:`aggregate`, which decides and labels its error: a trade's
cumulative P&L takes the stored paired error when every P&L date since inception carries it;
everything else — books, buckets and months over dates, a store written before the column — is
the root sum of squares of correlated errors, labelled "not the error of the cumulative P&L".

Checked by ``tests/test_backtest.py`` (a 5-date toy build calibrated once by the session fixture
``tests/_backtest_build.py::toy_backtest_build``; then, on copies, ``--no-calibrate`` shards
whose union is the unsharded store, ``--resume`` computing nothing, the config-hash refusal, the
projection printed first, stage 2 rendered by the runner with ``recalibrated`` false and the
desk sign, the attribution sum, the inception value against a fresh pricing, the VKO realised
state, the stability frame, the SSR reason, the incomplete date and its resume, stale inputs
(a calendar gap, moved manifest rates), both missing-close policies, the **walking test** of the
integrity invariant (every recorded input mutated in turn: the date and all later dates
unconfirmed, reported by stage 2), both refusals, the **crash matrix** of the storage (every
step of the attempt and pointer paths, three failure modes and real kills), concurrent commits
with ``gc``, the migration of the round-2 flat store and its crashes; and without the fixture:
the paired P&L and residual standard errors, the directional rates bucket on a non-parallel
curve move, parameter jitter, per-annum coupons, twin term sheets, the calibration charge, the
ξ₀ memo on float32 configs).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import functools
import hashlib
import io
import itertools
import json
import logging
import math
import os
import re
import shlex
import shutil
import socket
import stat
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Final, NoReturn, Protocol, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from volsto.calibration import guard
from volsto.calibration.cache import (
    LEVERAGE_NAME,
    XI0_MEMO,
    CacheMissError,
    LeverageCache,
    atomic_write,
    build_market,
    code_version,
    spec_key,
)
from volsto.calibration.fit_2f import (
    FIT_PRESETS,
    BreakEvenFitConfig,
    FitResult,
    fit_2f_marking,
    fit_preset,
)
from volsto.calibration.history import (
    DEFAULT_PILLARS,
    MIN_INCREMENTS,
    SurfaceHistory,
    SurfacePillarSource,
    hdn_available_dates,
)
from volsto.calibration.particle import CALIBRATION_CODE_TAG
from volsto.calibration.stability import PARAM_COLUMNS, flag_unidentified, rolling_fit
from volsto.config import (
    BergomiParams,
    CalibrationSpec,
    ConfigError,
    SimConfig,
    load_yaml,
    to_mapping,
)
from volsto.engine.grid import TimeGrid
from volsto.engine.mc import MonteCarlo
from volsto.market.compare import snapshot_difference
from volsto.market.curves import DiscountCurve, ForwardCurve
from volsto.market.import_hdn import (
    DEFAULT_CALENDAR_REPAIR,
    IMPORTER_TAG,
    import_day,
    write_snapshot,
)
from volsto.market.loaders import sabrw_fits_from_config, snapshot_spec, step0_source
from volsto.market.surface import ArbitrageError, ImpliedSurface, surface_from_config
from volsto.market.varswap import varswap_strike
from volsto.models.leverage import LeverageFunction
from volsto.models.lsv import LSV
from volsto.products.autocall import Autocall, Phoenix
from volsto.products.barrier import KnockInOption, KnockOutOption
from volsto.products.base import Product, daily_schedule
from volsto.products.cliquet import AdditiveCliquet
from volsto.products.conditional_variance import (
    SETTLEMENTS,
    DownVar,
    KnockOutVarianceSwap,
    UpVar,
)
from volsto.products.seasoning import TRADING_DAYS_PER_YEAR, RealisedHistory, Replay, replay
from volsto.products.variance import VarianceOption, VarianceSwap
from volsto.products.vko import VolKnockOutPut
from volsto.risk.attribution import (
    DETAILS,
    EXPLAIN_MODES,
    DryRunEngine,
    Explain,
    FrozenLeverageEngine,
    curve_move_sensitivities,
    explain,
    param_moved,
)
from volsto.risk.engine import LSVBuilder, RiskEngine, RiskState, Sensitivity, surface_of
from volsto.risk.greeks import delta_gamma, theta
from volsto.risk.greeks import vega as parallel_vega
from volsto.risk.ladders import K90, K110, curvature_T, skew_T, vega_T
from volsto.risk.volsto_sens import PARAMS, parameter_sensitivity
from volsto.studies import style
from volsto.studies.m6 import (
    AUTOCALL_BARRIER,
    AUTOCALL_COUPON,
    COUPON_BARRIER,
    KI_LEVEL,
    KI_PER_YEAR,
    PHOENIX_COUPON,
)
from volsto.studies.results import (
    DIMENSIONLESS,
    Column,
    FigureSpec,
    Results,
    ResultsBuilder,
    TableSpec,
)
from volsto.studies.runner import (
    MissingRequirements,
    Requirement,
    StrictLoader,
    StudyContext,
    git_state,
)
from volsto.viewers.grid import REPO_ROOT, parse_shard

if TYPE_CHECKING:
    from matplotlib.figure import Figure

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------------

#: One trading day (the seasoning date map and the attribution's time step).
TRADING_DT: Final[float] = 1.0 / TRADING_DAYS_PER_YEAR
#: Largest path count of the projection's pricing probe (the machine rule: tiny probes).
PROBE_PATHS_MAX: Final[int] = 20_000
#: The probe's second size is the first divided by this (the affine fit in paths).
PROBE_PATHS_RATIO: Final[int] = 5
#: Timings per probe pricing (the best is kept: another process's load only slows a run down).
PROBE_REPEATS: Final[int] = 2
#: Shard counts the projection always prints (plus the requested one).
SHARD_COUNTS: tuple[int, ...] = (1, 2, 4, 8)
#: Dates the probe tries before giving up (a failing date restarts its consecutive pair).
PROBE_ATTEMPTS: Final[int] = 6
#: Import seconds charged when the probe did not import (its snapshots existed) and no stored
#: timing exists: measured 2026-09-16 on 2022-07-27 (1.11 s, repair on, single process).
IMPORT_S_FALLBACK: Final[float] = 1.1
#: Vendors with an importer.
VENDORS: tuple[str, ...] = ("hdn",)
#: Rolling-book schedules and marking policies.
ROLLING_EVERY: tuple[str, ...] = ("month", "none")
ROLLING_MARK: tuple[str, ...] = ("daily", "inception")
#: Which trades the attribution explains.
ATTRIBUTION_TRADES: tuple[str, ...] = ("fixed", "all", "none")
#: The strike rule of the variance products (else a positive strike vol).
VS_STRIP: Final[str] = "vs_strip"
#: Kind-specific keys of a trade (besides ``id``, ``kind``, ``maturity``).
TRADE_KEYS: dict[str, tuple[str, ...]] = {
    "autocall": ("observations",),
    "phoenix": ("observations",),
    "cliquet": (),
    "vko_put": ("vol_ko", "moneyness"),
    "ko_var": ("barrier", "strike"),
    "var_swap": ("strike",),
    "uo_call": ("moneyness", "barrier"),
    "do_put": ("moneyness", "barrier"),
    "di_put": ("moneyness", "barrier"),
    "var_put": ("strike", "strike_ratio"),
    "up_var": ("barrier", "strike"),
    "down_var": ("barrier", "strike"),
}
#: Optional keys of a trade: ``ko_var.settlement`` (``maturity``, the default, or ``knock_out``,
#: the desk's convention); left out of the normalised mapping at its default, so a config without
#: it hashes exactly as before the key existed.
TRADE_OPTIONAL: dict[str, tuple[str, ...]] = {"ko_var": ("settlement",)}
#: The barrier-option kinds: (payoff, direction, knock) — daily-close monitoring, strict.
BARRIER_KINDS: dict[str, tuple[int, str, str]] = {
    "uo_call": (1, "up", "out"),
    "do_put": (-1, "down", "out"),
    "di_put": (-1, "down", "in"),
}
#: The variance kinds (struck at ``vs_strip`` or a strike vol, variance notional 1/(2 K_vol)).
VARIANCE_KINDS: tuple[str, ...] = ("ko_var", "var_swap", "var_put", "up_var", "down_var")
#: Each product's unit and scale: the study reports ``scale × value`` in that unit — % of
#: notional (notes, cliquet), % of the inception spot (VKO), vol points of vega notional
#: (variance swaps) — and never adds two units (P2).
TRADE_UNITS: dict[str, tuple[str, float]] = {
    "autocall": ("% notional", 100.0),
    "phoenix": ("% notional", 100.0),
    "cliquet": ("% notional", 100.0),
    "vko_put": ("% of inception spot", 100.0),
    "ko_var": ("vol pts (vega notional 1)", 100.0),
    "var_swap": ("vol pts (vega notional 1)", 100.0),
    "uo_call": ("% of inception spot", 100.0),
    "do_put": ("% of inception spot", 100.0),
    "di_put": ("% of inception spot", 100.0),
    "var_put": ("vol pts (vega notional 1)", 100.0),
    "up_var": ("vol pts (vega notional 1)", 100.0),
    "down_var": ("vol pts (vega notional 1)", 100.0),
}
#: Store layout.
HEADER_NAME = "backtest.json"
PROBE_NAME = "probe.json"
DATES_DIR = "dates"
ROWS_NAME = "rows.parquet"
FIT_NAME = "fit.json"
DONE_NAME = "done.json"
#: Storage (module docstring, *Storage*): per date the pointer, the attempts and the lock.
CURRENT_NAME = "CURRENT"
ATTEMPTS_DIR = "attempts"
STAGING_PREFIX = ".staging-"
POINTER_TMP_PREFIX = ".CURRENT.tmp-"
POINTER_VERSION: Final[int] = 1
#: How often a lock-free reader reads again an attempt that vanished while ``CURRENT`` changed
#: (with a short growing pause, :func:`_reread_pause`), before it reports the date as changing.
READ_RETRIES: Final[int] = 20
#: Hex digits of an attempt's content digest in its id.
ATTEMPT_DIGEST_CHARS: Final[int] = 32
MIGRATIONS_NAME = "migrations.json"
#: Where a migration moves flat files it could not put into an attempt (never deleted).
QUARANTINE_DIR = "quarantine"
#: The suffix a migrated flat-layout leftover gets before it is removed (so a crash during the
#: removal leaves a name the next open removes).
IMPORTED_SUFFIX = ".imported"
#: The import record next to each snapshot (:func:`bind_snapshot`).
IMPORT_SUFFIX = ".import.json"
#: The flat layout of volsto-backtest before attempts (migrated in place, :func:`migrate_store`):
#: ``dates/<date>/{rows.parquet, fit.json, done.json, failure.json}`` and ``<out>/.staging``.
FAILURE_NAME = "failure.json"
LEGACY_FILES: tuple[str, ...] = (ROWS_NAME, FIT_NAME, DONE_NAME, FAILURE_NAME)
LEGACY_STAGING_DIR = ".staging"
LEGACY_PREFIX = "legacy-"
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ISO_PREFIX = re.compile(r"(\d{4}-\d{2}-\d{2})\.(?:new|old)-")
_ATTEMPT_NAME = re.compile(rf"[a-z][a-z-]*-[0-9a-f]{{{ATTEMPT_DIGEST_CHARS}}}")
#: Verdict reasons other code reads.
NO_CURRENT = "no CURRENT"
LEGACY_REASON = "stored in the flat layout of an older volsto-backtest"
STALE_HASH = "stored under another config hash"
STALE_LEVERAGE = "its leverage is not in the configured cache"
#: Exit statuses of the CLI.
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
#: The attribution buckets stored on every row (``b_<name>``, dots as underscores), in order.
GREEK_BUCKETS: tuple[str, ...] = (
    "spot.delta",
    "spot.gamma",
    "rates.rho",
    "rates.repo",
    "surface.parallel_vega",
    "surface.vega_T",
    "surface.skew_T",
    "surface.curvature_T",
    *(f"params.{p}" for p in PARAMS),
    "time.decay",
    "time.carry",
)
BUCKETS: tuple[str, ...] = (
    *GREEK_BUCKETS,
    "recalibration",
    "residual",
    "cash_flows",
    "settlement",
    "unattributed",
)
#: The explain steps stored on every row (``s_<step>``: actual, stderr, explained).
STEPS: tuple[str, ...] = ("spot", "rates", "surface", "params", "factors", "time", "recalibration")
#: Start-of-period Greeks stored on attributed rows (``g_<name>`` and ``g_<name>_stderr``).
GREEKS: tuple[str, ...] = (
    "delta",
    "gamma",
    "vega",
    "rho",
    "repo",
    "theta",
    "theta_decay",
    "theta_carry",
    "theta_roll_down",
    *(f"d_{p}" for p in PARAMS),
)
#: Bucket groups of the tables (a bucket belongs to the first group whose prefix it has).
BUCKET_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("delta", ("spot.delta",)),
    ("gamma", ("spot.gamma",)),
    ("vega", ("surface.parallel_vega", "surface.vega_T")),
    ("skew", ("surface.skew_T",)),
    ("curvature", ("surface.curvature_T",)),
    ("params", tuple(f"params.{p}" for p in PARAMS)),
    ("rates", ("rates.rho", "rates.repo")),
    ("time", ("time.decay", "time.carry")),
    ("recalibration", ("recalibration",)),
    ("residual", ("residual",)),
    ("cash_flows", ("cash_flows",)),
    ("settlement", ("settlement",)),
    ("unattributed", ("unattributed",)),
)


def bucket_column(name: str) -> str:
    """``b_<bucket>`` with dots as underscores."""
    return "b_" + name.replace(".", "_")


#: Bucket sums whose paired standard error an attributed row stores (``grp_<name>_stderr``,
#: :attr:`~volsto.risk.attribution.Explain.group_stderrs`): the explained items of a step, the
#: time bucket (decay + carry), every Greek bucket, and those with the recalibration.
PAIRED_GROUPS: tuple[str, ...] = (
    "spot",
    "rates",
    "surface",
    "params",
    "time",
    "greeks",
    "explained",
)


def group_column(name: str) -> str:
    return f"grp_{name}_stderr"


def _row_columns() -> tuple[str, ...]:
    cols = [
        "date",
        "trade_id",
        "book",
        "kind",
        "inception",
        "age",
        "status",
        "reason",
        "unit",
        "scale",
        "value",
        "value_stderr",
        "flows",
        "flows_cum",
        "pnl",
        "pnl_stderr",
        "cum_pnl",
        "cum_pnl_stderr",
        "pnl_method",
        "pnl_note",
        "price_0",
        "price_0_stderr",
        "extra_pricings",
        "residual_paired",
    ]
    for b in BUCKETS:
        cols += [bucket_column(b), bucket_column(b) + "_stderr"]
    cols += [group_column(g) for g in PAIRED_GROUPS]
    for s in STEPS:
        cols += [f"s_{s}", f"s_{s}_stderr", f"s_{s}_explained"]
    for g in GREEKS:
        cols += [f"g_{g}", f"g_{g}_stderr"]
    cols += [
        "ladders_json",
        "realised_json",
        "realised_returns",
        "realised_vol",
        "vol_ko",
        "knocked_out",
        "knocked_in",
        "strike",
        "maturity",
        "gaps",
        "seconds",
        "config_hash",
    ]
    return tuple(cols)


#: Columns of ``rows.parquet`` (one row per (date, trade)).
ROW_COLUMNS: tuple[str, ...] = _row_columns()


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _jsonable(value: Any) -> Any:
    """``value`` as strict JSON data (NaN / inf → ``None``, numpy scalars and arrays → Python)."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (dt.date, pd.Timestamp)):
        return str(value)[:10]
    return str(value)


def _dumps(data: Any) -> str:
    return json.dumps(_jsonable(data), indent=1, sort_keys=True, allow_nan=False)


def _write_json(path: Path, data: Any) -> Path:
    text = _dumps(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dict(data) if isinstance(data, dict) else None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _resolve(p: str | Path) -> Path:
    """A config path: absolute as given, else under the repository root."""
    q = Path(p).expanduser()
    return q if q.is_absolute() else REPO_ROOT / q


def _display(p: str | Path) -> str:
    """``p`` repository-relative when inside the repository, else absolute."""
    resolved = Path(p).expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _cwd_path(p: str | Path) -> str:
    """``p`` for a command run from the current directory."""
    resolved = Path(p).expanduser().resolve()
    try:
        rel = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return str(resolved)
    return str(rel) if str(rel) != "." else "."


def _iso(value: Any, where: str) -> str:
    """An ISO date from a YAML date or string."""
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ConfigError(f"{where}: {value!r} is not a YYYY-MM-DD date") from exc
    raise ConfigError(f"{where}: expected a YYYY-MM-DD date, got {value!r}")


def _mapping(
    data: Any, keys: Sequence[str], where: str, optional: Sequence[str] = ()
) -> dict[str, Any]:
    """``data`` checked to be a mapping with exactly ``keys`` (and any of ``optional``)."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {data!r}")
    unknown = sorted(str(k) for k in set(data) - set(keys) - set(optional))
    missing = [k for k in keys if k not in data]
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}; expected {list(keys)}")
    if missing:
        raise ConfigError(f"{where}: missing keys {missing}")
    return dict(data)


def _num(value: Any, where: str, *, positive: bool = False, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    f = float(value)
    if not math.isfinite(f):
        raise ConfigError(f"{where}: must be finite")
    if positive and f <= 0:
        raise ConfigError(f"{where}: must be positive, got {value!r}")
    if integer and (not isinstance(value, int) or f != int(f)):
        raise ConfigError(f"{where}: expected an integer, got {value!r}")
    return f


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: expected true or false, got {value!r}")
    return value


def _choice(value: Any, choices: Sequence[str], where: str) -> str:
    if value not in choices:
        raise ConfigError(f"{where}: {value!r} must be one of {list(choices)}")
    return str(value)


def _trading_days(maturity: float, where: str) -> int:
    j = maturity * TRADING_DAYS_PER_YEAR
    n = round(j)
    if abs(j - n) > 1e-9 or n < 1:
        raise ConfigError(
            f"{where}: maturity {maturity:g} is not a whole number of trading days "
            f"({TRADING_DAYS_PER_YEAR} per year)"
        )
    return int(n)


# --------------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeSpec:
    """One term sheet of the book (module docstring; :data:`TRADE_KEYS`)."""

    id: str
    kind: str
    maturity: float
    observations: int | None = None
    vol_ko: float | None = None
    moneyness: float | None = None
    barrier: float | None = None
    strike: float | str | None = None
    strike_ratio: float | None = None
    settlement: str | None = None

    @classmethod
    def from_mapping(cls, data: Any, where: str) -> TradeSpec:
        if not isinstance(data, Mapping) or "kind" not in data:
            raise ConfigError(f"{where}: a trade is a mapping with a kind, got {data!r}")
        kind = _choice(data["kind"], tuple(TRADE_KEYS), f"{where}.kind")
        d = _mapping(
            data,
            ("id", "kind", "maturity", *TRADE_KEYS[kind]),
            where,
            optional=TRADE_OPTIONAL.get(kind, ()),
        )
        tid = d["id"]
        if not isinstance(tid, str) or not tid or not all(c.isalnum() or c in "_-" for c in tid):
            raise ConfigError(f"{where}.id: {tid!r} must match [A-Za-z0-9_-]+")
        maturity = _num(d["maturity"], f"{where}.maturity", positive=True)
        n_days = _trading_days(maturity, f"{where}.maturity")
        kw: dict[str, Any] = {}
        if kind in ("autocall", "phoenix"):
            n_obs = int(
                _num(d["observations"], f"{where}.observations", positive=True, integer=True)
            )
            if n_days % n_obs:
                raise ConfigError(
                    f"{where}: {n_obs} observations do not divide {n_days} trading days"
                )
            kw["observations"] = n_obs
        if kind == "cliquet" and abs(maturity * 12 - round(maturity * 12)) > 1e-9:
            raise ConfigError(f"{where}: the monthly cliquet needs a whole number of months")
        if kind == "vko_put":
            kw["vol_ko"] = _num(d["vol_ko"], f"{where}.vol_ko", positive=True)
            kw["moneyness"] = _num(d["moneyness"], f"{where}.moneyness", positive=True)
        if kind in ("ko_var", "up_var", "down_var"):
            kw["barrier"] = _num(d["barrier"], f"{where}.barrier", positive=True)
        if kind == "ko_var" and "settlement" in d:
            kw["settlement"] = _choice(d["settlement"], SETTLEMENTS, f"{where}.settlement")
        if kind in BARRIER_KINDS:
            kw["moneyness"] = _num(d["moneyness"], f"{where}.moneyness", positive=True)
            kw["barrier"] = _num(d["barrier"], f"{where}.barrier", positive=True)
            _, direction, _ = BARRIER_KINDS[kind]
            beyond = (
                kw["barrier"] > kw["moneyness"]
                if direction == "up"
                else (kw["barrier"] < kw["moneyness"])
            )
            if not beyond or (direction == "up") != (kw["barrier"] > 1.0):
                raise ConfigError(
                    f"{where}: a {kind} needs its barrier beyond the strike on the knock side and "
                    f"of the spot ({direction}), got barrier {kw['barrier']:g} and moneyness "
                    f"{kw['moneyness']:g}"
                )
        if kind == "var_put":
            kw["strike_ratio"] = _num(d["strike_ratio"], f"{where}.strike_ratio", positive=True)
        if kind in VARIANCE_KINDS:
            s = d["strike"]
            kw["strike"] = VS_STRIP if s == VS_STRIP else _num(s, f"{where}.strike", positive=True)
        return cls(tid, kind, maturity, **kw)

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": self.kind, "maturity": self.maturity}
        for k in TRADE_KEYS[self.kind]:
            out[k] = getattr(self, k)
        # an optional key at its default is left out (the hash of a config without it)
        if self.settlement is not None and self.settlement != "maturity":
            out["settlement"] = self.settlement
        return out

    @property
    def trading_days(self) -> int:
        return _trading_days(self.maturity, self.id)


#: The config's sections and their keys (strict).
CONFIG_SECTIONS: dict[str, tuple[str, ...]] = {
    "data": ("vendor", "root", "underlying", "missing_close"),
    "dates": ("start", "end"),
    "surface": ("essvi", "calendar_repair"),
    "marking": ("ssr_target", "skew_eps", "stage3"),
    "calibration": ("base_spec", "n_particles", "horizon"),
    "pricing": ("n_paths", "chunk_size", "seed"),
    "book": ("fixed", "rolling"),
    "attribution": ("mode", "detail", "trades", "pillars", "bump", "max_halvings"),
    "ssr": ("window", "pillars"),
    "stability": ("window_vol", "window_ssr", "share", "band", "fit"),
    "paths": ("out", "cache", "snapshots"),
}
#: Optional keys of a section.  ``marking.fit`` names the marking fit
#: (:data:`~volsto.calibration.fit_2f.FIT_PRESETS`); absent or ``m7`` it is the M7 fit and is left
#: out of the normalised mapping, so a config naming it hashes exactly as before the key existed.
CONFIG_OPTIONAL: dict[str, tuple[str, ...]] = {"marking": ("fit",)}
CONFIG_KEYS: tuple[str, ...] = ("name", *CONFIG_SECTIONS)
#: ``data.missing_close``: what an unreadable day file does (module docstring).
MISSING_CLOSE: tuple[str, ...] = ("fail", "skip_date")
#: ``stability.fit``: the keys of the historical-mode fit that must be given explicitly (any
#: other :class:`~volsto.calibration.fit_2f.BreakEvenFitConfig` field may be added).
STABILITY_FIT_REQUIRED: tuple[str, ...] = (
    "pillars",
    "mat_min",
    "k2",
    "skew_mode",
    "skew_eps",
    "nu_cap",
)
#: Keys left out of the content hash (module docstring).
HASH_EXCLUDED: tuple[tuple[str, str | None], ...] = (
    ("paths", None),
    ("data", "root"),
    ("dates", "end"),
)


@dataclass(frozen=True)
class BacktestConfig:
    """A backtest YAML, strictly validated (module docstring).  ``raw`` is the normalised
    mapping (what is hashed and stored); paths resolve against the repository root."""

    raw: Mapping[str, Any]
    source: str = ""
    fixed: tuple[TradeSpec, ...] = field(default=(), compare=False)
    rolling: tuple[TradeSpec, ...] = field(default=(), compare=False)

    # -- construction --------------------------------------------------------------------------

    @classmethod
    def from_mapping(cls, data: Any, *, source: str = "backtest config") -> BacktestConfig:
        top = _mapping(data, CONFIG_KEYS, source)
        name = top["name"]
        if not isinstance(name, str) or not name or not all(c.isalnum() or c in "_-" for c in name):
            raise ConfigError(f"{source}: name {name!r} must match [A-Za-z0-9_-]+")
        s = {
            k: _mapping(top[k], keys, f"{source}: {k}", CONFIG_OPTIONAL.get(k, ()))
            for k, keys in CONFIG_SECTIONS.items()
        }
        raw: dict[str, Any] = {"name": name}
        d = s["data"]
        raw["data"] = {
            "vendor": _choice(d["vendor"], VENDORS, "data.vendor"),
            "root": _path_str(d["root"], "data.root"),
            "underlying": _nonempty(d["underlying"], "data.underlying"),
            "missing_close": _choice(d["missing_close"], MISSING_CLOSE, "data.missing_close"),
        }
        start = _iso(s["dates"]["start"], "dates.start")
        end = _iso(s["dates"]["end"], "dates.end")
        if end < start:
            raise ConfigError(f"{source}: dates.end {end} precedes dates.start {start}")
        raw["dates"] = {"start": start, "end": end}
        essvi = _bool(s["surface"]["essvi"], "surface.essvi")
        repair = _bool(s["surface"]["calendar_repair"], "surface.calendar_repair")
        if repair and not essvi:
            raise ConfigError("surface.calendar_repair applies to eSSVI fits (essvi: true)")
        raw["surface"] = {"essvi": essvi, "calendar_repair": repair}
        m = s["marking"]
        if _bool(m["stage3"], "marking.stage3"):
            raise ConfigError(
                "marking.stage3: true is not supported by the backtest — stage 3 simulates each "
                "fit (minutes per date) and its assertion fails for the SPX (1.0, 0.10) mark "
                "(SPEC §15); the marked SSR is the fit's first-order value"
            )
        eps = _num(m["skew_eps"], "marking.skew_eps")
        if eps < 0:
            raise ConfigError("marking.skew_eps must be non-negative")
        raw["marking"] = {
            "ssr_target": _num(m["ssr_target"], "marking.ssr_target", positive=True),
            "skew_eps": eps,
            "stage3": False,
        }
        fit = _choice(m.get("fit", "m7"), sorted(FIT_PRESETS), "marking.fit")
        if fit != "m7":
            raw["marking"]["fit"] = fit
        c = s["calibration"]
        n_part = int(_num(c["n_particles"], "calibration.n_particles", positive=True, integer=True))
        if n_part % 2:
            raise ConfigError("calibration.n_particles must be even (antithetic particles)")
        horizon = _num(c["horizon"], "calibration.horizon", positive=True)
        raw["calibration"] = {
            "base_spec": _path_str(c["base_spec"], "calibration.base_spec"),
            "n_particles": n_part,
            "horizon": horizon,
        }
        p = s["pricing"]
        n_paths = int(_num(p["n_paths"], "pricing.n_paths", positive=True, integer=True))
        chunk = int(_num(p["chunk_size"], "pricing.chunk_size", positive=True, integer=True))
        seed = int(_num(p["seed"], "pricing.seed", integer=True))
        if n_paths % 2 or chunk % 2:
            raise ConfigError("pricing.n_paths and pricing.chunk_size must be even (antithetic)")
        if seed < 0:
            raise ConfigError("pricing.seed must be non-negative")
        raw["pricing"] = {"n_paths": n_paths, "chunk_size": chunk, "seed": seed}
        b = s["book"]
        if not isinstance(b["fixed"], list) or not b["fixed"]:
            raise ConfigError("book.fixed must be a non-empty list of trades")
        fixed = tuple(
            TradeSpec.from_mapping(t, f"book.fixed[{i}]") for i, t in enumerate(b["fixed"])
        )
        r = _mapping(b["rolling"], ("every", "mark", "products"), "book.rolling")
        every = _choice(r["every"], ROLLING_EVERY, "book.rolling.every")
        mark = _choice(r["mark"], ROLLING_MARK, "book.rolling.mark")
        if not isinstance(r["products"], list):
            raise ConfigError("book.rolling.products must be a list (empty with every: none)")
        rolling = tuple(
            TradeSpec.from_mapping(t, f"book.rolling.products[{i}]")
            for i, t in enumerate(r["products"])
        )
        if (every == "none") != (not rolling):
            raise ConfigError("book.rolling: every: none exactly when products is empty")
        for group, trades in (("fixed", fixed), ("rolling", rolling)):
            ids = [t.id for t in trades]
            if len(set(ids)) != len(ids):
                raise ConfigError(f"book.{group}: duplicate trade ids {ids}")
            for t in trades:
                if t.maturity > horizon + 1e-12:
                    raise ConfigError(
                        f"book.{group}.{t.id}: maturity {t.maturity:g} beyond the calibration "
                        f"horizon {horizon:g}"
                    )
        raw["book"] = {
            "fixed": [t.to_mapping() for t in fixed],
            "rolling": {
                "every": every,
                "mark": mark,
                "products": [t.to_mapping() for t in rolling],
            },
        }
        a = s["attribution"]
        pillars = _pillars(a["pillars"], "attribution.pillars")
        halvings = int(_num(a["max_halvings"], "attribution.max_halvings", integer=True))
        if halvings < 0:
            raise ConfigError("attribution.max_halvings must be non-negative")
        raw["attribution"] = {
            "mode": _choice(a["mode"], EXPLAIN_MODES, "attribution.mode"),
            "detail": _choice(a["detail"], DETAILS, "attribution.detail"),
            "trades": _choice(a["trades"], ATTRIBUTION_TRADES, "attribution.trades"),
            "pillars": list(pillars),
            "bump": _num(a["bump"], "attribution.bump", positive=True),
            "max_halvings": halvings,
        }
        if raw["attribution"]["mode"] != "sticky_leverage" and raw["attribution"]["trades"] != (
            "none"
        ):
            raise ConfigError(
                "attribution.mode: only sticky_leverage is supported by the backtest (the "
                "recalibrate mode calibrates every intermediate state: 11 to 48 calibrations "
                "per trade per date, SPEC §7.12.1)"
            )
        q = s["ssr"]
        window = int(_num(q["window"], "ssr.window", integer=True))
        if window < MIN_INCREMENTS:
            raise ConfigError(f"ssr.window must be at least {MIN_INCREMENTS} daily increments")
        ssr_pillars = _pillars(q["pillars"], "ssr.pillars")
        for t_ in ssr_pillars:
            if not any(abs(t_ - x) < 1e-9 for x in DEFAULT_PILLARS):
                raise ConfigError(
                    f"ssr.pillars: {t_:g} is not a history pillar {list(DEFAULT_PILLARS)}"
                )
        raw["ssr"] = {"window": window, "pillars": list(ssr_pillars)}
        st = s["stability"]
        wv = int(_num(st["window_vol"], "stability.window_vol", integer=True))
        ws = int(_num(st["window_ssr"], "stability.window_ssr", integer=True))
        if min(wv, ws) < MIN_INCREMENTS:
            raise ConfigError(f"stability windows must be at least {MIN_INCREMENTS}")
        share = _num(st["share"], "stability.share")
        if not 0 < share < 1:
            raise ConfigError("stability.share must lie in (0, 1)")
        fit_map = st["fit"]
        if not isinstance(fit_map, Mapping):
            raise ConfigError("stability.fit must be a mapping of BreakEvenFitConfig settings")
        missing_fit = [k for k in STABILITY_FIT_REQUIRED if k not in fit_map]
        if missing_fit:
            raise ConfigError(f"stability.fit: missing keys {missing_fit} (no silent default)")
        known = {f.name for f in dataclasses.fields(BreakEvenFitConfig)}
        unknown_fit = sorted(str(k) for k in set(fit_map) - known)
        if unknown_fit:
            raise ConfigError(f"stability.fit: unknown BreakEvenFitConfig keys {unknown_fit}")
        try:
            _fit_config(fit_map)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"stability.fit: {exc}") from exc
        raw["stability"] = {
            "window_vol": wv,
            "window_ssr": ws,
            "share": share,
            "band": _num(st["band"], "stability.band", positive=True),
            "fit": json.loads(_canonical(dict(fit_map))),
        }
        ps = s["paths"]
        raw["paths"] = {k: _path_str(ps[k], f"paths.{k}") for k in CONFIG_SECTIONS["paths"]}
        return cls(raw, source, fixed, rolling)

    # -- views ---------------------------------------------------------------------------------

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(_canonical(self.raw))  # type: ignore[no-any-return]

    def content(self) -> dict[str, Any]:
        """The hashed part of the mapping (module docstring): the mapping without
        :data:`HASH_EXCLUDED`, with ``calibration.base_spec`` replaced by the **content** that
        matters of that file (particle, simulation, local-vol and perturbation settings, the
        configured particles and horizon applied) and the resolved marking and stability fit
        configurations (library defaults included, so a changed default changes the hash)."""
        m = self.to_mapping()
        for section, key in HASH_EXCLUDED:
            if key is None:
                m.pop(section, None)
            else:
                m[section].pop(key, None)
        base = self.base_spec()
        m["calibration"]["base_spec"] = json.loads(
            json.dumps(
                {
                    "particle": to_mapping(base.particle),
                    "sim": to_mapping(base.sim),
                    "local_vol": to_mapping(base.local_vol),
                    "perturbation": to_mapping(base.perturbation),
                },
                sort_keys=True,
                default=str,
            )
        )
        m["marking"]["resolved"] = json.loads(
            json.dumps(to_mapping(self.fit_config()), sort_keys=True, default=str)
        )
        m["stability"]["resolved"] = json.loads(
            json.dumps(to_mapping(self.stability_fit_config()), sort_keys=True, default=str)
        )
        return m

    def content_hash(self) -> str:
        return hashlib.sha256(_canonical(self.content()).encode()).hexdigest()

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.raw[name]
        assert isinstance(value, Mapping)
        return value

    @property
    def name(self) -> str:
        return str(self.raw["name"])

    @property
    def data_root(self) -> Path:
        return _resolve(self.section("data")["root"])

    @property
    def underlying(self) -> str:
        return str(self.section("data")["underlying"])

    @property
    def start(self) -> str:
        return str(self.section("dates")["start"])

    @property
    def end(self) -> str:
        return str(self.section("dates")["end"])

    def path(self, key: str) -> Path:
        """``out`` / ``cache`` / ``snapshots`` resolved."""
        return _resolve(self.section("paths")[key])

    @property
    def n_paths(self) -> int:
        return int(self.section("pricing")["n_paths"])

    @property
    def n_particles(self) -> int:
        return int(self.section("calibration")["n_particles"])

    @property
    def horizon(self) -> float:
        return float(self.section("calibration")["horizon"])

    @property
    def attribution(self) -> Mapping[str, Any]:
        return self.section("attribution")

    @property
    def rolling_every(self) -> str:
        return str(self.section("book")["rolling"]["every"])

    @property
    def rolling_mark(self) -> str:
        return str(self.section("book")["rolling"]["mark"])

    def with_paths(
        self,
        *,
        out: str | Path | None = None,
        cache: str | Path | None = None,
        snapshots: str | Path | None = None,
    ) -> BacktestConfig:
        """The config with CLI path overrides (cwd-relative, stored absolute); a blank override
        is refused (it would resolve to the current directory)."""
        m = self.to_mapping()
        for key, value in (("out", out), ("cache", cache), ("snapshots", snapshots)):
            if value is not None:
                if not str(value).strip():
                    raise ConfigError(
                        f"--{key}: empty path (it would resolve to the current directory)"
                    )
                m["paths"][key] = str(Path(value).expanduser().absolute())
        return BacktestConfig.from_mapping(m, source=self.source)

    def base_spec(self) -> CalibrationSpec:
        """The base calibration spec with the configured particles and horizon (read once per
        config object)."""
        cached = self.__dict__.get("_base_spec")
        if isinstance(cached, CalibrationSpec):
            return cached
        c = self.section("calibration")
        try:
            base = load_yaml(_resolve(c["base_spec"]), CalibrationSpec)
        except OSError as exc:
            raise ConfigError(f"calibration.base_spec: {exc}") from exc
        particle = dataclasses.replace(
            base.particle, n_particles=int(c["n_particles"]), horizon=float(c["horizon"])
        )
        spec = dataclasses.replace(base, particle=particle)
        object.__setattr__(self, "_base_spec", spec)
        return spec

    @property
    def missing_close(self) -> str:
        return str(self.section("data")["missing_close"])

    def stability_fit_config(self) -> BreakEvenFitConfig:
        """The historical-mode fit of the stability flags (``stability.fit``)."""
        return _fit_config(self.section("stability")["fit"])

    def sim(self, spec: CalibrationSpec) -> SimConfig:
        """The pricing simulation: the spec's schedule and scheme (shared with the calibration,
        owner rule), the configured paths, chunk and seed."""
        p = self.section("pricing")
        return dataclasses.replace(
            spec.sim,
            n_paths=int(p["n_paths"]),
            chunk_size=int(p["chunk_size"]),
            seed=int(p["seed"]),
        )

    def fit_config(self) -> BreakEvenFitConfig:
        """The marking fit: the preset ``marking.fit`` (:data:`~volsto.calibration.fit_2f.
        FIT_PRESETS`; absent, the M7 fit) at ``marking.skew_eps``."""
        m = self.section("marking")
        return fit_preset(str(m.get("fit", "m7")), skew_eps=float(m["skew_eps"]))


def _fit_config(values: Mapping[str, Any]) -> BreakEvenFitConfig:
    """A :class:`BreakEvenFitConfig` from YAML values (lists become tuples)."""
    kw: dict[str, Any] = {str(k): tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return BreakEvenFitConfig(**kw)


def _path_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}: expected a path string, got {value!r}")
    return value


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _pillars(value: Any, where: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where}: expected a non-empty list of maturities")
    ps = tuple(_num(v, f"{where}[{i}]", positive=True) for i, v in enumerate(value))
    if any(b <= a for a, b in itertools.pairwise(ps)):
        raise ConfigError(f"{where}: maturities must be strictly increasing")
    return ps


def load_backtest_config(path: str | Path) -> BacktestConfig:
    """Strict YAML (:class:`~volsto.studies.runner.StrictLoader`) → :class:`BacktestConfig`."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{p}: {exc}") from exc
    try:
        data = yaml.load(text, Loader=StrictLoader)  # a SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: {exc}") from exc
    return BacktestConfig.from_mapping(data, source=_display(p))


# --------------------------------------------------------------------------------------------
# calendar, shards and the book
# --------------------------------------------------------------------------------------------


def calendar(cfg: BacktestConfig) -> list[str]:
    """The trading dates inside ``[start, end]``: the vendor's day files plus the dates its
    manifest lists as ``trading_days_missing`` (an expected trading day without a file must
    not silently shift the fixing grid: ``data.missing_close`` decides what it does).  A day
    that is neither present nor listed is invisible — the manifest is the only exchange
    calendar the importer has."""
    dates = set(hdn_available_dates(cfg.data_root))
    try:
        listed = json.loads(manifest_file(cfg).read_text(encoding="utf-8")).get(
            "trading_days_missing", []
        )
    except (DateFailure, OSError, ValueError, AttributeError):
        listed = []
    dates.update(str(d) for d in listed if isinstance(d, str))
    return sorted(d for d in dates if cfg.start <= d <= cfg.end)


def source_file(cfg: BacktestConfig, date: str) -> Path:
    return cfg.data_root / "day_by_date" / f"{date}_options.csv"


def shard_block(dates: Sequence[str], index: int, count: int) -> list[str]:
    """The ``index``-th of ``count`` contiguous blocks (1-based; ``numpy.array_split`` sizes)."""
    if count < 1 or not 1 <= index <= count:
        raise ValueError(f"shard must satisfy 1 <= i <= n, got {index}/{count}")
    bounds = np.array_split(np.arange(len(dates)), count)[index - 1]
    return [dates[int(i)] for i in bounds]


def rolling_inceptions(cfg: BacktestConfig, dates: Sequence[str]) -> list[str]:
    """The first calendar date of each month (``every: month``)."""
    if cfg.rolling_every == "none":
        return []
    seen: dict[str, str] = {}
    for d in dates:
        seen.setdefault(d[:7], d)
    return list(seen.values())


@dataclass(frozen=True)
class TradeInstance:
    """A trade of the backtest: a term sheet struck on ``inception``."""

    trade_id: str
    book: str
    spec: TradeSpec
    inception: str


def book_trades(cfg: BacktestConfig, dates: Sequence[str]) -> list[TradeInstance]:
    """The fixed trades (struck on the first date) and the rolling ones."""
    if not dates:
        return []
    out = [TradeInstance(f"fixed:{t.id}", "fixed", t, dates[0]) for t in cfg.fixed]
    for d in rolling_inceptions(cfg, dates):
        out += [TradeInstance(f"rolling:{d}:{t.id}", "rolling", t, d) for t in cfg.rolling]
    return out


def live_trades(
    cfg: BacktestConfig, calendar_dates: Sequence[str], trades: Sequence[TradeInstance], date: str
) -> list[TradeInstance]:
    """Trades struck on or before ``date`` whose last fixing is not before the previous date
    (settled-and-paid trades are dropped later, by the row)."""
    index = {d: i for i, d in enumerate(calendar_dates)}
    i = index[date]
    out = []
    for t in trades:
        j = index[t.inception]
        if j > i:
            continue
        if t.book == "rolling" and cfg.rolling_mark == "inception" and j != i:
            continue
        if i - j > t.spec.trading_days + 1:
            continue
        out.append(t)
    return out


def dependencies_of(
    cfg: BacktestConfig, calendar_dates: Sequence[str], trades: Sequence[TradeInstance], date: str
) -> tuple[list[str], list[str]]:
    """``(states, closes)``: the earlier dates whose **marked state** ``date``'s rows use — the
    previous calendar date (the engine's base: its state and leverage) and the inception date of
    every live trade (its strike, spot and discount curve) — and the earlier dates whose
    **close** they use — every date of a live trade's realised history, from its inception on."""
    cal = list(calendar_dates)
    i = cal.index(date)
    states = {cal[i - 1]} if i > 0 else set()
    starts = [i - 1] if i > 0 else []
    for t in live_trades(cfg, cal, trades, date):
        if t.inception != date:
            states.add(t.inception)
            starts.append(cal.index(t.inception))
    closes = cal[min(starts) : i] if starts else []
    return sorted(states), closes


def attributed(cfg: BacktestConfig, trade: TradeInstance) -> bool:
    rule = str(cfg.attribution["trades"])
    return rule == "all" or (rule == "fixed" and trade.book == "fixed")


@dataclass(frozen=True)
class BuiltTrade:
    """A fresh product and what its rows record."""

    product: Product
    unit: str
    scale: float
    strike: float
    vol_ko: float = float("nan")


def observation_times(spec: TradeSpec) -> np.ndarray:
    """``observations`` equally spaced dates on the trading grid (exact multiples of 1/252)."""
    assert spec.observations is not None
    n = spec.trading_days
    step = n // spec.observations
    return np.array([j / TRADING_DAYS_PER_YEAR for j in range(step, n + 1, step)])


def _annual(times: np.ndarray) -> bool:
    """Observation dates at 1, 2, …, n years exactly."""
    return bool(np.array_equal(times, np.arange(1, times.size + 1, dtype=np.float64)))


def build_product(
    spec: TradeSpec, spot: float, surface: ImpliedSurface, discount: DiscountCurve
) -> BuiltTrade:
    """The term sheet struck at ``spot`` on the inception ``surface`` (module docstring)."""
    unit, scale = TRADE_UNITS[spec.kind]
    T = spec.maturity
    daily = daily_schedule(T, TRADING_DAYS_PER_YEAR)
    if spec.kind == "autocall":
        obs = observation_times(spec)
        # per annum: c_i = c T_i (the M6 growing coupon i c when the dates are annual)
        coupons: float | tuple[float, ...] = (
            AUTOCALL_COUPON if _annual(obs) else tuple(AUTOCALL_COUPON * float(t) for t in obs)
        )
        prod: Product = Autocall(
            obs,
            discount,
            spot_reference=float(spot),
            coupons=coupons,
            ki_level=KI_LEVEL,
            ki_type="european",
            autocall_barriers=AUTOCALL_BARRIER,
            final_redemption="knock_in",
            notional=1.0,
        )
        return BuiltTrade(prod, unit, scale, float(spot))
    if spec.kind == "phoenix":
        obs = observation_times(spec)
        # per annum: the period coupon is c times the period length (c for annual periods)
        periods = np.diff(np.concatenate(([0.0], obs)))
        period_coupons: float | tuple[float, ...] = (
            PHOENIX_COUPON if _annual(obs) else tuple(PHOENIX_COUPON * float(d) for d in periods)
        )
        prod = Phoenix(
            obs,
            discount,
            spot_reference=float(spot),
            coupon=period_coupons,
            coupon_barrier=COUPON_BARRIER,
            memory=True,
            ki_level=KI_LEVEL,
            ki_type="american",
            ki_monitoring="discrete",
            ki_fixing_times=daily_schedule(float(obs[-1]), KI_PER_YEAR),
            autocall_barriers=AUTOCALL_BARRIER,
            final_redemption="knock_in",
            notional=1.0,
        )
        return BuiltTrade(prod, unit, scale, float(spot))
    if spec.kind == "cliquet":
        return BuiltTrade(AdditiveCliquet.study(T, discount), unit, scale, float("nan"))
    if spec.kind == "vko_put":
        assert spec.vol_ko is not None and spec.moneyness is not None
        k = spec.moneyness * float(spot)
        prod = VolKnockOutPut(k, T, spec.vol_ko, daily, discount, notional=1.0 / float(spot))
        return BuiltTrade(prod, unit, scale, k, spec.vol_ko)
    if spec.kind in BARRIER_KINDS:
        assert spec.moneyness is not None and spec.barrier is not None
        cp, direction, knock = BARRIER_KINDS[spec.kind]
        k = spec.moneyness * float(spot)
        cls = KnockOutOption if knock == "out" else KnockInOption
        prod = cls(
            k,
            T,
            cp,
            spec.barrier * float(spot),
            direction,
            discount,
            monitoring="discrete",
            fixing_times=daily,
            strict=True,
            notional=1.0 / float(spot),
        )
        return BuiltTrade(prod, unit, scale, k)
    k_vol = (
        math.sqrt(float(varswap_strike(surface, T)))
        if spec.strike == VS_STRIP
        else float(spec.strike)  # type: ignore[arg-type]
    )
    if spec.kind == "var_put":
        assert spec.strike_ratio is not None
        k_vol *= spec.strike_ratio
    notional = 1.0 / (2.0 * k_vol)
    if spec.kind == "ko_var":
        assert spec.barrier is not None
        prod = KnockOutVarianceSwap(
            daily,
            spec.barrier * float(spot),
            k_vol,
            discount,
            settlement=spec.settlement or "maturity",
            notional=notional,
        )
        return BuiltTrade(prod, unit, scale, k_vol)
    if spec.kind == "var_put":
        prod = VarianceOption(daily, k_vol, discount, cp=-1, notional=notional, annualisation=252.0)
        return BuiltTrade(prod, unit, scale, k_vol)
    if spec.kind in ("up_var", "down_var"):
        assert spec.barrier is not None
        maker = UpVar if spec.kind == "up_var" else DownVar
        prod = maker(
            daily,
            spec.barrier * float(spot),
            k_vol,
            discount,
            convention="corridor",
            notional=notional,
        )
        return BuiltTrade(prod, unit, scale, k_vol)
    prod = VarianceSwap(daily, k_vol * k_vol, discount, notional)
    return BuiltTrade(prod, unit, scale, k_vol)


# --------------------------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------------------------


#: Crash injection for the storage walking tests: called with the name of every step of the
#: attempt and pointer paths (:data:`CRASH_POINTS`); ``None`` in production.
CRASH_HOOK: Callable[[str], None] | None = None
#: The steps :data:`CRASH_HOOK` sees, in order.
CRASH_POINTS: tuple[str, ...] = (
    "stage.created",
    "stage.rows",
    "stage.fit",
    "stage.record",
    "stage.done",
    "stage.synced",
    "lock.acquired",
    "publish.renamed",
    "publish.synced",
    "pointer.decided",
    "pointer.written",
    "pointer.synced",
    "pointer.replaced",
    "pointer.dir_synced",
    "lock.released",
)


#: The steps of a migration :data:`CRASH_HOOK` sees (:func:`migrate_store`).
MIGRATION_CRASH_POINTS: tuple[str, ...] = (
    "migrate.journalled",
    "migrate.published",
    "migrate.pointed",
    "migrate.flat_removed",
    "migrate.reported",
    "migrate.leftover_marked",
    "migrate.leftover_removed",
    "migrate.bound",
)
#: Race injection for the storage tests: called with ``(what, date)`` right after a lock-free
#: reader read a date's ``CURRENT`` (``"pointer"``) and right after it read an attempt's files
#: (``"attempt"``) — where a concurrent writer or ``gc`` may act; ``None`` in production.
READ_HOOK: Callable[[str, str], None] | None = None


def _crash_point(name: str) -> None:
    if CRASH_HOOK is not None:
        CRASH_HOOK(name)


def _read_point(what: str, date: str) -> None:
    if READ_HOOK is not None:
        READ_HOOK(what, date)


def _fsync_path(path: Path) -> None:
    """``fsync`` a file or a directory (the durability of a rename is the directory's)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


#: REGULAR files every attempt check ignores (and only these): Finder's ``.DS_Store`` and the
#: AppleDouble companion ``._<name>`` of a regular file ``<name>`` the attempt holds (macOS
#: copies).  A directory (or any other non-regular entry) of those names is never ignored.
METADATA_FILE = ".DS_Store"
APPLEDOUBLE_PREFIX = "._"


def _ignored(name: str, names: Collection[str]) -> bool:
    return name == METADATA_FILE or (
        name.startswith(APPLEDOUBLE_PREFIX) and name[len(APPLEDOUBLE_PREFIX) :] in names
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_listing(directory: Path) -> dict[str, str]:
    """``{file name: SHA-256}`` of the regular files of a directory (sorted by name; the
    metadata files of :func:`_ignored` left out)."""
    return {n: _sha256(b) for n, b in read_files(directory).items()}


def read_files(directory: Path) -> dict[str, bytes]:
    """Every regular file of a directory, read once (metadata files left out; a subdirectory is
    listed as ``<name>/`` with empty content, so it can never pass as an attempt file).  Raises
    ``OSError`` when the directory or a file vanishes while it is read."""
    entries = sorted(os.listdir(directory))
    paths = {n: directory / n for n in entries}
    names = {n for n, q in paths.items() if q.is_file()}
    out: dict[str, bytes] = {}
    for n, q in paths.items():
        if n not in names:
            out[f"{n}/"] = b""  # a non-regular entry is never ignored
            continue
        if _ignored(n, names):
            continue
        with open(q, "rb") as fh:
            out[n] = fh.read()
    return out


def listing_digest(listing: Mapping[str, str]) -> str:
    return hashlib.sha256(_canonical(dict(sorted(listing.items()))).encode()).hexdigest()


def attempt_id(label: str, listing: Mapping[str, str]) -> str:
    """``<label>-<digest>``: an attempt is addressed by the SHA-256 of its file listing, so any
    attempt verifies on its own (:meth:`Attempt.verify`)."""
    return f"{label}-{listing_digest(listing)[:ATTEMPT_DIGEST_CHARS]}"


@dataclass(frozen=True)
class Attempt:
    """One immutable outcome of a date: ``dates/<date>/attempts/<id>/`` holding ``done.json``
    (its status, record and bookkeeping) and, for results and skips, ``rows.parquet`` and
    ``fit.json``."""

    date: str
    id: str
    path: Path

    @property
    def digest(self) -> str:
        return self.id.rsplit("-", 1)[-1]

    def read(self) -> AttemptContent:
        """Every file, read once (raises ``OSError`` when the attempt vanishes meanwhile)."""
        content = AttemptContent(self, read_files(self.path))
        _read_point("attempt", self.date)
        return content

    def listing(self) -> dict[str, str]:
        return self.read().listing

    def verify(
        self, listing: Mapping[str, str] | None = None, content: AttemptContent | None = None
    ) -> str:
        """``""`` when the files (``content``, else read now) hash to the id and to ``listing``
        when given, else why not."""
        if content is None:
            try:
                content = self.read()
            except FileNotFoundError:
                return f"its attempt {self.id} is missing"
            except OSError as exc:
                return f"its attempt {self.id} is unreadable ({type(exc).__name__}: {exc})"
        now = content.listing
        if listing is not None:
            why = _listing_mismatch(dict(listing), now)
            if why:
                return why
        if listing_digest(now)[:ATTEMPT_DIGEST_CHARS] != self.digest:
            return f"the files of its attempt {self.id} do not hash to its id"
        return ""

    def doc(self) -> dict[str, Any] | None:
        """``done.json``, unverified (display only: decisions read :class:`AttemptContent`)."""
        return _read_json(self.path / DONE_NAME)

    def created(self) -> str:
        return str((self.doc() or {}).get("created_utc", ""))


@dataclass(frozen=True)
class AttemptContent:
    """The files of an attempt as one read saw them: a verdict checks these bytes against the
    recorded hashes and parses the same bytes (never the files again)."""

    attempt: Attempt
    files: Mapping[str, bytes]

    @property
    def name(self) -> str:
        return self.attempt.id

    @property
    def listing(self) -> dict[str, str]:
        return {n: _sha256(b) for n, b in sorted(self.files.items())}

    def json(self, name: str) -> dict[str, Any]:
        """A JSON file of the attempt (``KeyError`` when absent, ``ValueError`` when not an
        object)."""
        data = json.loads(self.files[name].decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{name} is not a JSON object")
        return data

    def frame(self) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.files[ROWS_NAME]))


def _listing_mismatch(expected: Mapping[str, str], now: Mapping[str, str]) -> str:
    for name in sorted(set(expected) | set(now)):
        if expected.get(name) == now.get(name):
            continue
        if name not in now:
            return f"its {name} is missing"
        if name not in expected:
            return f"its attempt holds an unexpected file {name}"
        return f"its {name} was modified"
    return ""


class BacktestStore:
    """The per-date store (module docstring, *Storage*): ``<out>/backtest.json``,
    ``<out>/probe.json``, ``<out>/migrations.json`` and per date ``dates/<date>/CURRENT`` (the
    pointer) and ``dates/<date>/attempts/<id>/`` (immutable outcomes); the locks are ``flock``s
    on the date directories and the store root (:meth:`lock`, :meth:`store_lock`).

    Writes: an outcome is staged in ``attempts/.staging-*``, published by ONE rename to
    ``attempts/<id>`` and pointed to by ONE atomic replace of ``CURRENT``, both under the date's
    lock; nothing a writer publishes is modified or removed afterwards (``gc`` removes
    non-current attempts on request).  Reads never lock: ``CURRENT`` is always a complete old or
    new pointer."""

    def __init__(self, root: Path) -> None:
        # resolved once: every process started before a swap of a symlinked root keeps working
        # on (and locking) the same directories
        self.root = Path(root).resolve()

    @property
    def dates_root(self) -> Path:
        return self.root / DATES_DIR

    def date_dir(self, date: str) -> Path:
        return self.dates_root / date

    def attempts_dir(self, date: str) -> Path:
        return self.date_dir(date) / ATTEMPTS_DIR

    def pointer_path(self, date: str) -> Path:
        return self.date_dir(date) / CURRENT_NAME

    def header(self) -> dict[str, Any] | None:
        return _read_json(self.root / HEADER_NAME)

    def write_header(self, cfg: BacktestConfig) -> None:
        _write_json(
            self.root / HEADER_NAME,
            {
                "name": cfg.name,
                "config": cfg.to_mapping(),
                "config_hash": cfg.content_hash(),
                "config_path": cfg.source,
                "created_utc": _utc_now(),
                "code_version": code_version(),
            },
        )

    def dates(self) -> list[str]:
        """The dates with a directory in the store."""
        if not self.dates_root.is_dir():
            return []
        return sorted(p.name for p in self.dates_root.iterdir() if _ISO_DATE.fullmatch(p.name))

    # -- reading (never locks) -------------------------------------------------------------------

    def pointer(self, date: str) -> dict[str, Any] | None:
        """The parsed ``CURRENT`` (``None`` without one, ``{"invalid": reason}`` if unusable)."""
        return parse_pointer(self.pointer_bytes(date))

    def pointer_bytes(self, date: str) -> bytes | None:
        """``CURRENT``'s bytes (``None`` without one)."""
        try:
            return self.pointer_path(date).read_bytes()
        except FileNotFoundError:
            return None

    def attempt(self, date: str, attempt: str) -> Attempt:
        return Attempt(date, attempt, self.attempts_dir(date) / attempt)

    def attempts(self, date: str) -> list[Attempt]:
        """The published attempts of a date (staging directories excluded), oldest first by
        their unverified creation time (callers decide on verified content)."""
        root = self.attempts_dir(date)
        if not root.is_dir():
            return []
        found = [
            self.attempt(date, p.name)
            for p in root.iterdir()
            if p.is_dir() and _ATTEMPT_NAME.fullmatch(p.name)
        ]
        return sorted(found, key=lambda a: (a.created(), a.id))

    def current(self, date: str) -> Attempt | None:
        """The attempt ``CURRENT`` names, unverified (projections and messages only; results are
        read through :class:`Ledger`)."""
        ptr = self.pointer(date)
        if ptr is None or "invalid" in ptr:
            return None
        return self.attempt(date, str(ptr["attempt"]))

    def current_fit(self, date: str) -> dict[str, Any] | None:
        """The current fit record, unverified (the cost projection only)."""
        cur = self.current(date)
        return None if cur is None else _read_json(cur.path / FIT_NAME)

    @property
    def quarantine_root(self) -> Path:
        return self.root / QUARANTINE_DIR

    def legacy_files(self, date: str) -> list[Path]:
        """Files of the flat layout of volsto-backtest before attempts (``migrate`` moves them)."""
        d = self.date_dir(date)
        return [d / n for n in LEGACY_FILES if (d / n).is_file()]

    def leftovers(self) -> dict[str, list[Path]]:
        """Staging directories and pointer temporaries of interrupted (or running) writers, and
        a staging root of the flat layout, by kind."""
        out: dict[str, list[Path]] = {"staging": [], "pointer": [], "flat": []}
        for d in self.dates():
            a = self.attempts_dir(d)
            if a.is_dir():
                out["staging"] += sorted(a.glob(f"{STAGING_PREFIX}*"))
            out["pointer"] += sorted(self.date_dir(d).glob(f"{POINTER_TMP_PREFIX}*"))
        legacy_staging = self.root / LEGACY_STAGING_DIR
        if legacy_staging.is_dir():
            out["flat"] += sorted(legacy_staging.iterdir())
        return out

    @contextlib.contextmanager
    def locks(self, dates: Iterable[str]) -> Iterator[dict[str, DateLock]]:
        """The locks of several dates, taken in date order (writers hold one date lock at a
        time, so this never deadlocks with them)."""
        with contextlib.ExitStack() as stack:
            yield {d: stack.enter_context(self.lock(d)) for d in sorted(set(dates))}

    # -- writing ---------------------------------------------------------------------------------

    @contextlib.contextmanager
    def lock(self, date: str) -> Iterator[DateLock]:
        """The date's exclusive lock: ``fcntl.flock`` on the date DIRECTORY itself, released by
        the kernel when the holder dies (macOS and Linux lock directory descriptors alike).

        Why this is sound where a lock file is not: ``flock`` excludes the holders of one inode.
        A lock file's path can be unlinked or replaced while held, so a later locker gets its
        own inode — and checking ``fstat``/``stat`` after locking does not help, since a
        newcomer that creates the replacement sees them agree.  Here the locked inode is the
        directory that holds the data, and every mutation made under the lock goes through the
        locked descriptor (:class:`DateLock`: ``dir_fd``-relative renames, replaces, unlinks and
        removals), never through the path again.  A process that resolves a swapped path locks
        and writes a different directory, so two holders never mutate one directory; readers
        take no lock and verify what they read by hash.  The store root is resolved once, when
        the store object is made."""
        import fcntl

        d = self.date_dir(date)
        d.mkdir(parents=True, exist_ok=True)
        fd = os.open(d, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        held = DateLock(date, fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield held
        finally:
            held.close()  # closing the descriptor releases the lock

    @contextlib.contextmanager
    def store_lock(self) -> Iterator[int]:
        """The store-wide exclusive lock (``flock`` on the store root directory): migrations
        run under it, date locks nested inside in date order."""
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield fd
        finally:
            os.close(fd)

    def stage(self, date: str) -> Path:
        """A new empty staging directory ``attempts/.staging-<host>-<pid>-<random>``."""
        root = self.attempts_dir(date)
        root.mkdir(parents=True, exist_ok=True)
        prefix = f"{STAGING_PREFIX}{_host_tag()}-{os.getpid()}-"
        return Path(tempfile.mkdtemp(prefix=prefix, dir=root))

    def publish(self, held: DateLock, staged: Path, label: str) -> Attempt:
        """Publish a complete staged directory as ``attempts/<label>-<digest>`` (one rename,
        relative to the locked date directory; an attempt with the same content already
        published is kept and the staged copy dropped)."""
        attempt = self.attempt(held.date, attempt_id(label, file_listing(staged)))
        afd = held.attempts_fd()
        try:
            os.rename(staged.name, attempt.id, src_dir_fd=afd, dst_dir_fd=afd)
        except OSError:
            if not _exists_at(attempt.id, afd):
                raise
            shutil.rmtree(staged.name, dir_fd=afd, ignore_errors=True)  # identical content
        _crash_point("publish.renamed")
        os.fsync(afd)
        _crash_point("publish.synced")
        return attempt

    def set_pointer(
        self, held: DateLock, attempt: Attempt, status: str, listing: Mapping[str, str]
    ) -> None:
        """Point ``CURRENT`` at a published attempt: the pointer (version, attempt id, status,
        the SHA-256 of every file as the caller verified them) is written to a temporary file,
        synced, and moved over ``CURRENT`` by one ``os.replace`` — all relative to the locked
        date directory, which is then synced.  A ``CURRENT`` of another pointer version is
        refused (:class:`RefusedError`), never overwritten."""
        if attempt.date != held.date:
            raise ValueError(f"{attempt.id} is not an attempt of {held.date}")
        foreign = _foreign_version(held.read_pointer())
        if foreign:
            raise RefusedError(f"{held.date}: {foreign}")
        data = {
            "version": POINTER_VERSION,
            "date": attempt.date,
            "attempt": attempt.id,
            "status": status,
            "files": dict(listing),
            "written_utc": _utc_now(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        name = f"{POINTER_TMP_PREFIX}{os.getpid()}-{os.urandom(6).hex()}"
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=held.fd)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(_dumps(data))
                _crash_point("pointer.written")
                fh.flush()
                os.fsync(fh.fileno())
            _crash_point("pointer.synced")
            os.replace(name, CURRENT_NAME, src_dir_fd=held.fd, dst_dir_fd=held.fd)
        except (Exception, KeyboardInterrupt):
            with contextlib.suppress(OSError):
                os.unlink(name, dir_fd=held.fd)
            raise
        _crash_point("pointer.replaced")
        os.fsync(held.fd)
        _crash_point("pointer.dir_synced")

    def remove_attempt(self, held: DateLock, attempt: Attempt) -> None:
        """Remove a non-current attempt (``gc`` only), relative to the locked date directory.  A
        symbolic link is never followed or removed."""
        if attempt.date != held.date:
            raise ValueError(f"{attempt.id} is not an attempt of {held.date}")
        ptr = parse_pointer(held.read_pointer())
        if ptr is None or "invalid" in ptr or ptr.get("attempt") == attempt.id:
            raise ValueError(f"{held.date}: {attempt.id} is current (or CURRENT is unusable)")
        afd = held.attempts_fd()
        if stat.S_ISLNK(os.stat(attempt.id, dir_fd=afd, follow_symlinks=False).st_mode):
            raise ValueError(f"{held.date}: {attempt.id} is a symbolic link")
        shutil.rmtree(attempt.id, dir_fd=afd)


@dataclass
class DateLock:
    """A held date lock: the flock'ed descriptor of ``dates/<date>/``, through which every
    mutation of the date goes (:meth:`BacktestStore.lock`)."""

    date: str
    fd: int
    _attempts: int | None = None

    def attempts_fd(self) -> int:
        """The descriptor of ``attempts/`` inside the locked directory (created if needed)."""
        if self._attempts is None:
            with contextlib.suppress(FileExistsError):
                os.mkdir(ATTEMPTS_DIR, 0o755, dir_fd=self.fd)
            self._attempts = os.open(
                ATTEMPTS_DIR, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0), dir_fd=self.fd
            )
        return self._attempts

    def read_pointer(self) -> bytes | None:
        """``CURRENT`` as seen through the locked directory."""
        try:
            fd = os.open(CURRENT_NAME, os.O_RDONLY, dir_fd=self.fd)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as fh:
            return fh.read()

    def still_at(self, path: Path) -> bool:
        """Whether ``path`` (followed through symbolic links) is still the locked directory.  A
        sanity check that narrows, not closes, the window of an outside move of a live store."""
        try:
            st = os.stat(path)
        except OSError:  # gone, not a directory, unreadable, a symlink loop: not the same
            return False
        fs = os.fstat(self.fd)
        return (st.st_dev, st.st_ino) == (fs.st_dev, fs.st_ino)

    def names(self) -> list[str]:
        return sorted(os.listdir(self.fd))

    def attempt_names(self) -> list[str]:
        return sorted(os.listdir(self.attempts_fd())) if ATTEMPTS_DIR in self.names() else []

    def unlink(self, name: str) -> None:
        os.unlink(name, dir_fd=self.fd)

    def release_attempts(self) -> None:
        """Close the ``attempts/`` descriptor (a store-wide command holds one lock per date:
        one descriptor each is enough)."""
        if self._attempts is not None:
            os.close(self._attempts)
            self._attempts = None

    def close(self) -> None:
        self.release_attempts()
        os.close(self.fd)


def _exists_at(name: str, dir_fd: int) -> bool:
    try:
        os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _foreign_version(data: bytes | None) -> str:
    """``""`` unless ``data`` is a ``CURRENT`` written by another pointer version (a refusal)."""
    ptr = parse_pointer(data)
    if ptr is not None and ptr.get("foreign_version") is not None:
        return str(ptr["invalid"])
    return ""


def parse_pointer(data: bytes | None) -> dict[str, Any] | None:
    """A ``CURRENT`` pointer from its bytes (``None`` for none; ``{"invalid": reason}`` when it
    is not a pointer this code reads, an unknown :data:`POINTER_VERSION` included)."""
    if data is None:
        return None
    try:
        ptr = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        return {"invalid": f"not JSON ({exc})"}
    if not isinstance(ptr, dict) or not isinstance(ptr.get("files"), dict):
        return {"invalid": "not a pointer"}
    if ptr.get("version") != POINTER_VERSION:
        return {
            "invalid": f"CURRENT has pointer version {ptr.get('version')!r}, written by another "
            f"volsto-backtest (this code reads version {POINTER_VERSION}): refused, never "
            "overwritten",
            "foreign_version": ptr.get("version"),
        }
    if not _ATTEMPT_NAME.fullmatch(str(ptr.get("attempt", ""))):
        return {"invalid": f"names no attempt ({ptr.get('attempt')!r})"}
    return ptr


def _host_tag() -> str:
    return re.sub(r"[^A-Za-z0-9]", "", socket.gethostname())[:24] or "host"


def _staging_owner_alive(path: Path) -> bool:
    """Whether the process that created a staging directory (by its name) still runs here."""
    parts = path.name[len(STAGING_PREFIX) :].split("-")
    if len(parts) < 3 or parts[0] != _host_tag():
        return True  # another host (or an unknown name): assume alive
    try:
        os.kill(int(parts[1]), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError):
        return True
    return True


class RefusedError(RuntimeError):
    """The store or the cache refuses the run (exit 2): a changed config hash, a missing
    leverage under ``--no-calibrate``."""


def manifest_file(cfg: BacktestConfig) -> Path:
    """The vendor manifest the importer reads (its Treasury curve gives every date's rates):
    ``day_by_date/manifest.json``, else ``manifest.json`` at the data root (the importer's
    search order)."""
    for cand in (
        cfg.data_root / "day_by_date" / "manifest.json",
        cfg.data_root / "manifest.json",
    ):
        if cand.is_file():
            return cand
    raise DateFailure(f"no manifest.json under {cfg.data_root}")


#: The checksum of a calendar date whose day file does not exist.
ABSENT = "absent"


class InputIndex:
    """Checksums of the vendor inputs, computed once per process: each day file, each date's
    **manifest entry** (what the importer reads of ``manifest.json`` for that date: its rate
    curve ``rates[date]``, the ``files`` entry of its day file and ``product`` — a manifest that
    gains a new day leaves older dates valid), and per date the **inputs digest**: the SHA-256
    chain over every ``(date, day-file checksum, manifest-entry checksum)`` of the calendar up to
    that date.  A date reads its own file and manifest entry and the closes of every earlier
    date, so the digest changes when any of them does, and when a date appears in or disappears
    from the calendar prefix."""

    def __init__(self, cfg: BacktestConfig, vendor_dates: Sequence[str]) -> None:
        self.cfg = cfg
        self.dates = list(vendor_dates)
        self._sha: dict[str, str] = {}
        self._manifest: dict[str, Any] | None = None
        self._entries: dict[str, str] = {}
        self._digests: dict[str, str] = {}

    def file_sha(self, date: str) -> str:
        """The day file's SHA-256, or ``"absent"`` for a calendar date without a file."""
        if date not in self._sha:
            src = source_file(self.cfg, date)
            self._sha[date] = _file_sha256(src) if src.is_file() else ABSENT
        return self._sha[date]

    def manifest_sha(self, date: str) -> str:
        """The SHA-256 of the manifest entry the import of ``date`` reads."""
        if date not in self._entries:
            if self._manifest is None:
                try:
                    data = json.loads(manifest_file(self.cfg).read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise DateFailure(f"unreadable vendor manifest: {exc}") from exc
                self._manifest = dict(data) if isinstance(data, dict) else {}
            m = self._manifest
            name = source_file(self.cfg, date).name
            entry = {
                "product": m.get("product"),
                "rates": (m.get("rates") or {}).get(date),
                "file": next((f for f in m.get("files", []) if f.get("name") == name), None),
            }
            self._entries[date] = hashlib.sha256(_canonical(_jsonable(entry)).encode()).hexdigest()
        return self._entries[date]

    def digest(self, date: str) -> str:
        if date not in self._digests:
            h = hashlib.sha256(b"volsto-backtest inputs").hexdigest()
            for d in self.dates:
                link = f"{h}|{d}:{self.file_sha(d)}:{self.manifest_sha(d)}"
                h = hashlib.sha256(link.encode()).hexdigest()
                self._digests[d] = h
                if d == date:
                    break
        if date not in self._digests:
            raise KeyError(f"{date} is not a vendor date of the configured range")
        return self._digests[date]


# --------------------------------------------------------------------------------------------
# integrity: the dependency record (module docstring, *Integrity*)
# --------------------------------------------------------------------------------------------

#: Layout version of the dependency record this code writes (2: the leverage content digest),
#: and the versions it verifies (a record of any other version is stale).
RECORD_VERSION: Final[int] = 2
RECORD_VERSIONS: tuple[int, ...] = (1, 2)
#: What a version-1 record with results cannot confirm (:attr:`Verdict.legacy_unverified`).
LEGACY_UNVERIFIED = "leverage content not verified (record version 1)"
#: What recomputing such dates does (``status`` and study.md say it with the command).
LEGACY_RECOMPUTE = (
    "recomputing them records version 2: their rows and standard errors are recomputed under "
    "the current code, and a leverage taken from the cache records no calibration (their "
    "calibration count drops)"
)
#: The one cause a ``skip_date`` skip may record (re-derived by every verdict).
SKIP_CAUSE = "absent day file"
#: The link of a dependency whose marked state was unavailable to the date that recorded it.
UNAVAILABLE = "unavailable"
#: Stored statuses whose date directory holds results (rows and fit).
RESULT_STATUSES: tuple[str, ...] = ("ok", "incomplete")
#: The only line of a snapshot a re-import of the same inputs changes.
_CREATED_LINE = re.compile(rb"^  created_utc: .*\r?\n", re.MULTILINE)


def snapshot_digest(path: Path) -> str:
    """SHA-256 of a snapshot file without its ``provenance.created_utc`` line (``"absent"``
    without a file): the digest of the market a date was marked from.  The import is
    deterministic — two imports of 2022-10-28 differ in that line only (measured 2026-09-16) —
    so shards importing the same day concurrently record one digest whichever write lands
    last, while any edit of the market, the surface or the provenance changes it."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ABSENT
    return hashlib.sha256(_CREATED_LINE.sub(b"", data)).hexdigest()


def bytes_digest(data: bytes | None) -> str:
    """A file's SHA-256 from its bytes, ``"absent"`` without the file."""
    return ABSENT if data is None else _sha256(data)


def snapshot_bytes_digest(data: bytes | None) -> str:
    """:func:`snapshot_digest` of bytes already read (``"absent"`` for none)."""
    return ABSENT if data is None else _sha256(_CREATED_LINE.sub(b"", data))


def rows_fingerprint(data: bytes | None) -> dict[str, Any]:
    """``{"sha256", "n"}`` of a ``rows.parquet`` from its bytes (``n`` ``None`` when absent or
    unreadable)."""
    n: int | None = None
    if data is not None:
        try:
            n = int(pq.read_metadata(io.BytesIO(data)).num_rows)
        except Exception:  # unreadable bytes: their checksum differs from the recorded one anyway
            n = None
    return {"sha256": bytes_digest(data), "n": n}


def leverage_content_digest(path: Path | IO[bytes]) -> str:
    """SHA-256 of the NUMERIC content of a ``leverage.npz``: the sorted ``(name, dtype, shape,
    bytes)`` of every member but ``metadata`` (whose creation time and commit change on every
    recalibration that reproduces the same numbers).  Raises on an unreadable archive."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for name in sorted(z.files):
            if name == "metadata":
                continue
            arr = np.ascontiguousarray(z[name])
            h.update(f"{name}|{arr.dtype.str}|{arr.shape}|".encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def state_params(params: Any) -> dict[str, float]:
    """The fitted model parameters as the fit record stores them (:data:`PARAMS`)."""
    return {
        k: float(params[k] if isinstance(params, Mapping) else getattr(params, k)) for k in PARAMS
    }


def state_link(
    date: str,
    snapshot: str,
    params: Mapping[str, float],
    key: str,
    content: str | None = None,
    *,
    version: int = RECORD_VERSION,
) -> str:
    """What a later date consumes of ``date``: its marked state — the snapshot digest, the fitted
    parameters, the leverage key and (record version 2) the leverage content digest — as one
    SHA-256.  Version 1 links (stores written before the content digest) omit the content."""
    payload: dict[str, Any] = {
        "date": date,
        "snapshot": snapshot,
        "params": state_params(params),
        "key": key,
    }
    if version >= 2:
        payload["leverage_content"] = content
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


@dataclass(frozen=True)
class Verdict:
    """The integrity verdict of one date (:meth:`Ledger.verdict`), from one consistent read."""

    date: str
    status: str
    """``done`` (an ``ok`` outcome whose ``CURRENT`` pointer and dependency record verify, its
    dependencies' verdicts included), ``incomplete``, ``failed``, ``skipped`` (the same for
    those outcomes), ``pending`` (it verifies, but a date it depends on is not stored yet — a
    shard boundary or an ``--only-dates`` window — so it cannot be confirmed), ``unsettled`` (a
    consistent read could not be obtained while other writers changed the store: unknown, never
    acted on), ``stale`` (anything else; :attr:`reason` says what) or ``missing`` (no
    ``CURRENT``)."""
    reason: str = ""
    doc: Mapping[str, Any] | None = None
    """The confirmed (or pending) outcome: the current attempt's ``done.json``."""
    attempt: Attempt | None = None
    """The attempt judged (the current one, or a candidate)."""
    links: Mapping[int, str] = field(default_factory=dict)
    """The date's state links by record version (:func:`state_link`) when confirmed."""
    leverage: bool = False
    """Whether the date's leverage is complete in the configured cache (confirmed dates)."""
    waits: tuple[str, ...] = ()
    """The unstored dates a ``pending`` verdict waits for."""
    origin: tuple[str, str] | None = None
    """``(date, reason)`` where a ``stale`` verdict's mismatch originates (itself, or the stale
    date it depends on)."""
    stored: Mapping[str, Any] | None = None
    """The attempt's ``done.json`` whenever its files verify, even when its record no longer
    does (what the refusals and the pointer rule read); for a date still in the flat layout, its
    flat ``done.json`` (so the refusals see it before any migration)."""
    content: AttemptContent | None = None
    """The verified bytes of the attempt (set with :attr:`stored`): what stage 2 reads."""
    pointer_sha256: str = ""
    """The SHA-256 of the ``CURRENT`` bytes this verdict read."""
    note: str = ""
    """Display only (never a classification): e.g. a later failed attempt, from verified
    bytes."""
    flat: bool = False
    """The date is still in the flat layout (it needs :func:`migrate_store`)."""
    foreign: bool = False
    """Its ``CURRENT`` was written by another pointer version (a refusal for every writer)."""
    legacy_unverified: str = ""
    """Non-empty (:data:`LEGACY_UNVERIFIED`) for an outcome with results whose record is of
    version 1: it verifies, but the numbers of its own leverage were never recorded, so only
    the key and its completeness are checked — reported by ``status`` and the study, never
    silent (the owner's decision for the 25-date proof of concept, 2026-09-17)."""

    @property
    def link(self) -> str:
        return self.links.get(RECORD_VERSION, UNAVAILABLE)

    @property
    def settled(self) -> bool:
        """Nothing to compute (``--resume`` skips it)."""
        return self.status in ("done", "skipped")

    @property
    def has_results(self) -> bool:
        """Its rows and fit are confirmed (``done`` or ``incomplete``)."""
        return self.status in ("done", "incomplete")

    @property
    def results_bearing(self) -> bool:
        """The attempt's files verify and hold results (``ok`` / ``incomplete``), whatever its
        record says now."""
        return self.stored is not None and self.stored_status in RESULT_STATUSES

    @property
    def files(self) -> Path:
        """The directory of the verified attempt."""
        if self.attempt is None or self.stored is None:
            raise ValueError(f"{self.date}: no verified attempt ({self.status})")
        return self.attempt.path

    @property
    def verified(self) -> AttemptContent:
        if self.content is None:
            raise ValueError(f"{self.date}: no verified attempt ({self.status})")
        return self.content

    @property
    def stored_status(self) -> str:
        return str((self.stored or {}).get("status", ""))

    @property
    def stored_hash(self) -> str:
        return str((self.stored or {}).get("config_hash", ""))

    @property
    def created(self) -> str:
        return str((self.stored or {}).get("created_utc", ""))


def computed_under_another_config(v: Verdict, config_hash: str) -> bool:
    """**The one judgement** of "this date's stored outcome was computed under another config":
    its verified (or flat) ``done.json`` names another config hash.  Used by the ``--force``
    refusal of ``run`` and by the commands stage 2 prints."""
    return v.stored is not None and v.stored_hash != config_hash


def verdict_rank(v: Verdict) -> int:
    """How much an attempt protects: ``done`` 5, a pending ``ok`` 4, ``incomplete`` 3, a pending
    ``incomplete`` 2, any other attempt whose files verify and hold results (stale ones included)
    1, anything without results (failed, skipped, unverifiable, missing) 0."""
    if v.status == "done":
        return 5
    if v.status == "pending":
        return {"ok": 4, "incomplete": 2}.get(v.stored_status, 0)
    if v.status == "incomplete":
        return 3
    return 1 if v.results_bearing else 0


def _record_version(v: Verdict) -> object:
    record = (v.stored or {}).get("record")
    return record.get("version") if isinstance(record, Mapping) else None


def choose_pointer(
    current: Verdict,
    attempts: Sequence[Verdict],
    config_hash: str,
    new: Verdict | None = None,
) -> Verdict | None:
    """**The pointer rule** — the attempt ``CURRENT`` should name, or ``None`` to keep it.

    Nothing moves while any verdict involved is ``unsettled``.  Among the published attempts
    (``attempts``, the new one included) only those of this config are eligible.  The new
    attempt is taken when it protects at least as much as the current one and as much as any
    eligible attempt (:func:`verdict_rank`); otherwise the best eligible attempt (the newest
    among equals) is taken when it protects more than the current one.  An attempt without
    results never becomes current while any attempt of the date — current or not, of any
    config — holds verified results.  Once the date has an attempt recorded at
    :data:`RECORD_VERSION`, an attempt of an older record version is never made current (it
    stays done only while it already is: its evidence is weaker, never a fallback)."""
    if current.status == "unsettled" or any(v.status == "unsettled" for v in attempts):
        return None
    has_current_record = any(_record_version(v) == RECORD_VERSION for v in (current, *attempts))
    eligible = [
        v
        for v in attempts
        if v.attempt is not None
        and not computed_under_another_config(v, config_hash)
        and v.stored is not None
        and not (has_current_record and _record_version(v) != RECORD_VERSION)
    ]
    if not eligible:
        return None
    rank_now = verdict_rank(current)
    best = max(verdict_rank(v) for v in eligible)
    choice: Verdict | None = None
    if (
        new is not None
        and new in eligible
        and verdict_rank(new) >= rank_now
        and verdict_rank(new) == best
    ):
        choice = new
    elif best > rank_now:
        choice = max(
            (v for v in eligible if verdict_rank(v) == best),
            key=lambda v: (v.created, v.attempt.id if v.attempt else ""),
        )
    if choice is None:
        return None
    any_results = current.results_bearing or any(v.results_bearing for v in attempts)
    if any_results and not choice.results_bearing:
        return None
    return choice


def importer_tag_reason(tag: str | None) -> str:
    """Why a date whose snapshot was imported under another importer tag is stale."""
    return (
        f"its snapshot was imported by importer {tag or 'untagged (before 2026-09-22)'} "
        f"(current {IMPORTER_TAG}): --resume re-imports it and recomputes the date"
    )


def _record_mismatch(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> str:
    """Why a stored dependency record differs from the recomputed one (``""`` when equal)."""
    for name in dict.fromkeys([*expected, *stored]):
        a, b = stored.get(name), expected.get(name)
        if a == b:
            continue
        if name == "config_hash":
            return STALE_HASH
        if name == "inputs_digest":
            return "its inputs changed (a day file up to it, the manifest or the calendar)"
        if name == "calibration_code_tag":
            return f"calibration code tag {a} (now {b})"
        if name == "snapshot":
            if b == ABSENT:
                return "its snapshot is missing"
            return "its snapshot differs from the one it was computed from"
        if name == "cause":
            return f"the cause of its skip no longer holds ({b})"
        if name in ("rows", "fit"):
            file = ROWS_NAME if name == "rows" else FIT_NAME
            now = b if isinstance(b, Mapping) else {"sha256": b}
            was = a if isinstance(a, Mapping) else {"sha256": a}
            if now.get("sha256") == ABSENT:
                return f"its {file} is missing"
            if now.get("n") != was.get("n"):
                return f"its {file} holds {now.get('n')} row(s), {was.get('n')} recorded"
            return f"its {file} was modified"
        if name == "leverage" and isinstance(a, Mapping) and isinstance(b, Mapping):
            if a.get("key") != b.get("key"):
                if str(b.get("key", "")).startswith("unrecoverable"):
                    return f"its leverage key cannot be recomputed ({b.get('key')})"
                return "its leverage key no longer follows from its snapshot, fit and base spec"
            if not b.get("complete"):
                return STALE_LEVERAGE
            if a.get("content") != b.get("content"):
                return "its leverage's numbers differ from the ones it was priced with"
            return "its leverage is now in the configured cache"
        if name == "previous" and isinstance(a, Mapping) and isinstance(b, Mapping):
            if a.get("date") != b.get("date"):
                return f"its previous calendar date is now {b.get('date')} (was {a.get('date')})"
            state = "present" if b.get("leverage") else "missing"
            return f"the leverage of its previous date {b.get('date')} is now {state}"
        if name in ("links", "closes") and isinstance(a, Mapping) and isinstance(b, Mapping):
            what = "marked state" if name == "links" else "close (snapshot)"
            for e in sorted(set(a) | set(b)):
                if a.get(e) != b.get(e):
                    if e not in a or e not in b:
                        return f"its dependencies changed ({e})"
                    if b.get(e) == ABSENT:
                        return f"the snapshot of {e}, whose close it uses, is missing"
                    return f"the {what} of {e}, which it depends on, changed"
        return f"its record differs in {name}"
    return ""


class _Reread(Exception):  # noqa: N818 — control flow of the lock-free reader
    """An attempt vanished while it was read and ``CURRENT`` changed meanwhile: read again."""


def _reread_pause(k: int) -> None:
    time.sleep(min(0.05, 0.001 * 2**k))


@dataclass(frozen=True)
class SnapshotRead:
    """A snapshot as one read saw it: its digest, the spec parsed from the same bytes, and the
    importer tag its provenance names (``None`` when untagged: imported before 2026-09-22)."""

    digest: str
    spec: CalibrationSpec | None
    error: str = ""
    importer_tag: str | None = None


def read_snapshot(base: CalibrationSpec, path: Path) -> tuple[SnapshotRead, bytes | None]:
    """Read a snapshot ONCE: its :func:`snapshot_digest` and the spec parsed from those bytes
    (through a private copy, so a concurrent re-import cannot mix two versions)."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return SnapshotRead(ABSENT, None, "absent"), None
    digest = snapshot_bytes_digest(data)
    try:
        with tempfile.TemporaryDirectory(prefix="volsto-snap-") as tmp:
            copy = Path(tmp) / path.name
            copy.write_bytes(data)
            spec = snapshot_spec(base, copy)
    except (OSError, KeyError, TypeError, ValueError, ConfigError, yaml.YAMLError) as exc:
        return SnapshotRead(digest, None, f"{type(exc).__name__}: {exc}"), data
    return SnapshotRead(digest, spec, importer_tag=_importer_tag_of(data)), data


def _importer_tag_of(data: bytes) -> str | None:
    """``provenance.importer_tag`` of a snapshot's bytes (``None`` when absent or unreadable)."""
    try:
        prov = yaml.safe_load(data).get("provenance")
    except (AttributeError, ValueError, yaml.YAMLError):
        return None
    tag = prov.get("importer_tag") if isinstance(prov, Mapping) else None
    return str(tag) if tag is not None else None


class RecordInputs(Protocol):
    """Where :func:`dependency_record` reads the snapshot digests and the leverage."""

    def snapshot(self, date: str) -> str: ...

    def leverage(self, date: str, params: Any, version: int) -> dict[str, Any]: ...


class Ledger:
    """The one reader of a backtest store (module docstring, *Integrity*).

    :meth:`verdict` is the one answer to "is this date done": ``status``, ``run --resume``
    (its selection and its check before each date), both refusals of ``run``, the pointer rule
    (:func:`choose_pointer`), ``gc`` and the study's requirements and reads all use it.  It reads
    the date's ``CURRENT`` and every file of the attempt it names ONCE, checks those bytes
    against the pointer's hashes and parses the same bytes — never the files again — and then
    recomputes the attempt's dependency record with :func:`dependency_record`, the function the
    run wrote it with.  An attempt that vanishes while ``CURRENT`` changes is read again; when
    that keeps happening the verdict is ``unsettled``."""

    def __init__(
        self,
        cfg: BacktestConfig,
        store: BacktestStore,
        inputs: InputIndex,
        *,
        snapshots: Path,
        cache: LeverageCache,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.inputs = inputs
        self.snapshots = snapshots
        self.cache = cache
        self.hash = cfg.content_hash()
        self.vendor_calendar = list(inputs.dates)
        self._base: CalibrationSpec | None = None
        self._verdicts: dict[str, Verdict] = {}
        self._calendar: list[str] | None = None
        self._trades: list[TradeInstance] | None = None
        self._keys: dict[tuple[str, str], str] = {}
        self._skip_cache: dict[str, Verdict] | None = None
        self._snapshots: dict[str, SnapshotRead] = {}
        self._leverage: dict[tuple[str, int, int, int], tuple[bool, str | None]] = {}

    @classmethod
    def of(cls, cfg: BacktestConfig, *, snapshots: Path | None = None) -> Ledger:
        """The ledger of ``cfg``'s store, cache and snapshots (its ``paths``)."""
        return cls(
            cfg,
            BacktestStore(cfg.path("out")),
            InputIndex(cfg, calendar(cfg)),
            snapshots=snapshots if snapshots is not None else cfg.path("snapshots"),
            cache=LeverageCache(cfg.path("cache")),
        )

    @property
    def base(self) -> CalibrationSpec:
        if self._base is None:
            self._base = self.cfg.base_spec()
        return self._base

    def snapshot_path(self, date: str) -> Path:
        return self.snapshots / f"{self.cfg.underlying.lower()}_{date}.yaml"

    def inputs_digest(self, date: str) -> str:
        try:
            return self.inputs.digest(date)
        except (DateFailure, KeyError, OSError) as exc:
            return f"unavailable: {exc}"

    def invalidate(self, date: str | None = None) -> None:
        """Forget the verdicts from ``date`` on, the calendar and the skips (a commit of ``date``
        can change every later verdict); without a date, everything, this pass's snapshot reads
        included (a commit never changes a snapshot; an import invalidates everything)."""
        for d in list(self._verdicts):
            if date is None or d >= date:
                del self._verdicts[d]
        self._calendar = None
        self._trades = None
        self._skip_cache = None
        if date is None:
            self._snapshots.clear()

    # -- what a record reads (RecordInputs) ------------------------------------------------------

    def read_snapshot(self, date: str) -> SnapshotRead:
        """The date's snapshot, read once per pass (:func:`read_snapshot`)."""
        hit = self._snapshots.get(date)
        if hit is None:
            hit = self._snapshots[date] = read_snapshot(self.base, self.snapshot_path(date))[0]
        return hit

    def snapshot(self, date: str) -> str:
        return self.read_snapshot(date).digest

    def leverage_state(self, key: str) -> tuple[bool, str | None]:
        """``(complete, content digest)`` of a cache entry, memoised by its file identity (the
        one completeness test, :meth:`~volsto.calibration.cache.LeverageCache.has_key`, then
        :func:`leverage_content_digest`)."""
        path = self.cache.root / key / LEVERAGE_NAME
        try:
            st_ = path.stat()
        except OSError:
            return False, None
        memo = (key, st_.st_ino, st_.st_mtime_ns, st_.st_size)
        hit = self._leverage.get(memo)
        if hit is None:
            complete = self.cache.has_key(key)
            content: str | None = None
            if complete:
                try:
                    content = leverage_content_digest(path)
                except Exception:  # an archive that opens but does not load is incomplete
                    complete = False
            hit = self._leverage[memo] = (complete, content)
        return hit

    def leverage(self, date: str, params: Any, version: int = RECORD_VERSION) -> dict[str, Any]:
        """``{"key", "complete"[, "content"]}``: the leverage key recomputed from the date's
        snapshot (as :meth:`read_snapshot` parsed it), the fitted parameters and the base spec
        (memoised: content-addressed), whether the configured cache holds it complete and
        (record version 2) the digest of its numbers."""
        snap = self.read_snapshot(date)
        try:
            if snap.spec is None:
                raise ValueError(snap.error or "no snapshot")
            memo = (snap.digest, _canonical(state_params(params)))
            key = self._keys.get(memo)
            if key is None:
                spec = dataclasses.replace(snap.spec, model=BergomiParams(**state_params(params)))
                key = self._keys[memo] = spec_key(spec)
        except (KeyError, TypeError, ValueError, ConfigError) as exc:
            out: dict[str, Any] = {"key": f"unrecoverable: {type(exc).__name__}", "complete": False}
            if version >= 2:
                out["content"] = None
            return out
        complete, content = self.leverage_state(key)
        out = {"key": key, "complete": complete}
        if version >= 2:
            out["content"] = content
        return out

    # -- the effective calendar ----------------------------------------------------------------

    def skips(self) -> dict[str, str]:
        """The vendor dates dropped by a confirmed ``skipped`` outcome (``skip_date``)."""
        return {d: str((v.stored or {}).get("error", "")) for d, v in self._skip_verdicts().items()}

    def _skip_verdicts(self) -> dict[str, Verdict]:
        """A date is skipped when one of its skip attempts — the current one or, since a skip
        never displaces results (:func:`choose_pointer`), a newer one next to them — verifies:
        its files hash to its address, its record (inputs digest included) recomputes equal, and
        its cause — the day file is absent — still holds.  A skip record has no dependencies,
        so this needs no calendar."""
        if self.cfg.missing_close != "skip_date":
            return {}
        if self._skip_cache is not None:
            return self._skip_cache
        out: dict[str, Verdict] = {}
        for d in self.vendor_calendar:
            skips = [
                a
                for a in self.store.attempts(d)
                if a.id.rsplit("-", 1)[0] in ("skipped", f"{LEGACY_PREFIX}skipped")
            ]
            try:  # the one cause a skip may have: the day file is absent (re-derived here)
                if skips and self.inputs.file_sha(d) != ABSENT:
                    continue
            except (DateFailure, OSError):
                continue
            for a in reversed(skips):
                try:
                    content = a.read()
                except OSError:
                    continue
                if a.verify(None, content):
                    continue
                try:
                    doc = content.json(DONE_NAME)
                    if doc.get("status") != "skipped" or doc.get("date") != d:
                        continue
                    mismatch = self._check(d, doc, content)[0]
                except Exception as exc:
                    mismatch = f"{type(exc).__name__}: {exc}"
                if not mismatch:
                    out[d] = Verdict(
                        d, "skipped", _outcome_reason(doc), doc, a, stored=doc, content=content
                    )
                    break
        self._skip_cache = out
        return out

    def calendar(self) -> list[str]:
        """The vendor calendar without the confirmed skips."""
        if self._calendar is None:
            skipped = self.skips()
            self._calendar = [d for d in self.vendor_calendar if d not in skipped]
        return self._calendar

    def trades(self) -> list[TradeInstance]:
        if self._trades is None:
            self._trades = book_trades(self.cfg, self.calendar())
        return self._trades

    def previous(self, date: str) -> str | None:
        cal = self.calendar()
        if date not in cal:
            return None
        i = cal.index(date)
        return cal[i - 1] if i > 0 else None

    def dependencies(self, date: str) -> tuple[list[str], list[str]]:
        """:func:`dependencies_of` on the effective calendar."""
        cal = self.calendar()
        return dependencies_of(self.cfg, cal, self.trades(), date) if date in cal else ([], [])

    # -- verdicts ------------------------------------------------------------------------------

    def _read_current(
        self, date: str
    ) -> tuple[bytes | None, Attempt | None, AttemptContent | None, str]:
        """``(CURRENT bytes, attempt, content, reason)`` from one read of the pointer and of
        every file of the attempt it names (``reason`` ``""`` when the bytes hash as recorded).
        Raises :class:`_Reread` when the attempt vanished while ``CURRENT`` changed."""
        data = self.store.pointer_bytes(date)
        _read_point("pointer", date)
        ptr = parse_pointer(data)
        if ptr is None:
            return None, None, None, NO_CURRENT
        if "invalid" in ptr:
            return data, None, None, f"its {CURRENT_NAME} is unusable ({ptr['invalid']})"
        att = self.store.attempt(date, str(ptr["attempt"]))
        try:
            content = att.read()
        except OSError as exc:
            if self.store.pointer_bytes(date) != data:
                raise _Reread(date) from exc
            if isinstance(exc, FileNotFoundError) and not att.path.exists():
                return data, att, None, f"its attempt {att.id} is missing"
            return data, att, None, f"its attempt is unreadable ({type(exc).__name__}: {exc})"
        why = att.verify(ptr["files"], content)
        if why == f"its {ROWS_NAME} was modified":
            try:  # done.json sorts first, so it verified: its record's row count is trustworthy
                recorded = (content.json(DONE_NAME).get("record") or {}).get("rows") or {}
                now = rows_fingerprint(content.files.get(ROWS_NAME))["n"]
                why += f" ({now} row(s), {recorded.get('n')} recorded)"
            except (KeyError, ValueError):
                pass
        if why:
            return data, att, None, why
        try:
            doc = content.json(DONE_NAME)
        except (KeyError, ValueError, UnicodeDecodeError):
            return data, att, None, f"its {DONE_NAME} is unreadable"
        if doc.get("status") != ptr.get("status") or doc.get("date") != date:
            return data, att, None, f"its {CURRENT_NAME} does not match its {DONE_NAME}"
        return data, att, content, ""

    def verify_current(self, date: str) -> tuple[Attempt | None, dict[str, Any] | None, str]:
        """``(attempt, done.json, reason)`` of the current attempt from one consistent read
        (``reason`` ``"no CURRENT"`` without a pointer)."""
        for k in range(READ_RETRIES):
            try:
                _, att, content, why = self._read_current(date)
            except _Reread:
                _reread_pause(k)
                continue
            return att, (None if content is None else content.json(DONE_NAME)), why
        return None, None, f"its {CURRENT_NAME} kept changing while it was read"

    def verdict(self, date: str) -> Verdict:
        """The date's verdict (memoised; the earlier calendar dates are judged first, in order,
        since a date's verdict needs its dependencies')."""
        hit = self._verdicts.get(date)
        if hit is not None:
            return hit
        cal = self.calendar()
        if date in cal:
            for d in cal[: cal.index(date)]:
                if d not in self._verdicts:
                    self._verdicts[d] = self._judge(d)
        v = self._verdicts[date] = self._judge(date)
        return v

    def verdicts(self, dates: Sequence[str] | None = None) -> dict[str, Verdict]:
        return {d: self.verdict(d) for d in (self.vendor_calendar if dates is None else dates)}

    def candidate(self, date: str, attempt: Attempt) -> Verdict:
        """The verdict ``attempt`` would have as the date's current attempt (not memoised), from
        one read of its files verified against its content address."""
        content, doc, why = self._read_candidate(date, attempt)
        if why or content is None or doc is None:
            return Verdict(date, "stale", why, attempt=attempt, origin=(date, why))
        return self._attempt_verdict(date, content, doc)

    def _read_candidate(
        self, date: str, attempt: Attempt
    ) -> tuple[AttemptContent | None, dict[str, Any] | None, str]:
        try:
            content = attempt.read()
        except OSError as exc:
            return None, None, f"its attempt {attempt.id} is unreadable ({type(exc).__name__})"
        why = attempt.verify(None, content)
        if why:
            return None, None, why
        try:
            doc = content.json(DONE_NAME)
        except (KeyError, ValueError, UnicodeDecodeError):
            return None, None, f"its {DONE_NAME} is unreadable"
        if doc.get("date") != date:
            return None, None, f"its {DONE_NAME} names another date"
        return content, doc, ""

    def attempt_verdicts(self, date: str, current: Verdict) -> list[Verdict]:
        """The candidate verdicts of every published attempt of a date (the verified attempt
        ``current`` judged reused)."""
        out = []
        for a in self.store.attempts(date):
            if (
                current.stored is not None
                and current.attempt is not None
                and a.id == current.attempt.id
            ):
                out.append(current)
            else:
                out.append(self.candidate(date, a))
        return out

    def _judge(self, date: str) -> Verdict:
        for k in range(READ_RETRIES):
            try:
                return self._judge_once(date)
            except _Reread:
                _reread_pause(k)
                continue
        why = f"its {CURRENT_NAME} kept changing while it was read ({READ_RETRIES} reads)"
        return Verdict(date, "unsettled", why, origin=(date, why))

    def _judge_once(self, date: str) -> Verdict:
        data, att, content, why = self._read_current(date)
        sha = "" if data is None else _sha256(data)
        if self.cfg.missing_close == "skip_date" and date not in self.calendar():
            skip = self._skip_verdicts().get(date)
            if skip is not None:
                return dataclasses.replace(skip, pointer_sha256=sha)
        if why == NO_CURRENT:
            if self.store.legacy_files(date):
                return self._flat_verdict(date)
            n = len(self.store.attempts(date))
            if n == 0:
                reason = "not computed"
            elif foreign_pointer_refusal(self.store):  # no writer of this version may adopt
                reason = f"no {CURRENT_NAME} ({n} published attempt(s))"
            else:
                reason = (
                    f"no {CURRENT_NAME} ({n} published attempt(s): "
                    f"{backtest_command(self.cfg, 'run', [date])} adopts the best confirmed one)"
                )
            return Verdict(date, "missing", reason)
        if why or content is None:
            return Verdict(
                date,
                "stale",
                why,
                attempt=att,
                origin=(date, why),
                pointer_sha256=sha,
                foreign=bool(_foreign_version(data)),
            )
        v = self._attempt_verdict(date, content, content.json(DONE_NAME))
        return dataclasses.replace(v, pointer_sha256=sha, note=self._later_failure(date, v))

    def _later_failure(self, date: str, v: Verdict) -> str:
        """A display note about a failed attempt newer than the current one — from VERIFIED
        bytes only (its files hash to its address)."""
        current = v.attempt.id if v.attempt is not None else ""
        failed = [
            a
            for a in self.store.attempts(date)
            if a.id != current and a.id.rsplit("-", 1)[0] in ("failed", f"{LEGACY_PREFIX}failed")
        ]
        best: tuple[str, str] | None = None
        for a in failed:
            _, doc, why = self._read_candidate(date, a)
            if why or doc is None or doc.get("status") != "failed":
                continue
            created = str(doc.get("created_utc", ""))
            if created > v.created and (best is None or created > best[0]):
                best = (created, str(doc.get("error", "")))
        return "" if best is None else f"a later attempt failed: {best[1]}"

    def _flat_verdict(self, date: str) -> Verdict:
        """A date still in the flat layout: ``stale`` (it needs the migration), with its flat
        outcome judged as the attempt the migration would publish — so the refusals see its
        results and its config before anything is migrated."""
        stored: dict[str, Any] | None = None
        try:
            data = (self.store.date_dir(date) / DONE_NAME).read_bytes()
            doc = json.loads(data.decode("utf-8"))
            if isinstance(doc, dict) and doc.get("date") == date:
                stored = doc
        except (OSError, ValueError, UnicodeDecodeError):
            stored = None
        header = self.store.header() or {}
        # the config the outcome names (its own done.json), else the store header's
        owner = str((stored or {}).get("config_hash") or header.get("config_hash") or "")
        if owner and owner != self.hash:
            reason = (
                f"{LEGACY_REASON}, under config hash {owner[:12]}: migrate it with that config "
                f"(this config hashes to {self.hash[:12]})"
            )
        else:
            reason = (
                f"{LEGACY_REASON}: {backtest_command(self.cfg, 'migrate')} moves it into attempts"
            )
        return Verdict(date, "stale", reason, origin=(date, reason), stored=stored, flat=True)

    def _attempt_verdict(self, date: str, content: AttemptContent, doc: dict[str, Any]) -> Verdict:
        att = content.attempt
        try:
            why, waits, root, unsettled = self._check(date, doc, content)
        except _Reread:
            raise
        except Exception as exc:  # never a crash: an unusable record is a mismatch
            why = f"its record cannot be checked: {type(exc).__name__}: {exc}"
            waits, root, unsettled = (), None, False
        if unsettled:
            return Verdict(
                date, "unsettled", why, attempt=att, origin=root, stored=doc, content=content
            )
        if why:
            origin = root if root is not None else (date, why)
            return Verdict(
                date, "stale", why, attempt=att, origin=origin, stored=doc, content=content
            )
        status = str(doc["status"])
        links, leverage = self._own_links(date, doc, content)
        record = doc.get("record") or {}
        legacy = (
            LEGACY_UNVERIFIED if record.get("version") == 1 and status in RESULT_STATUSES else ""
        )
        if waits:
            reason = f"waits for {', '.join(waits)} (not stored yet); stored as {status}"
            return Verdict(
                date,
                "pending",
                reason,
                doc,
                att,
                links,
                leverage,
                waits,
                stored=doc,
                content=content,
                legacy_unverified=legacy,
            )
        state = "done" if status == "ok" else status
        return Verdict(
            date,
            state,
            _outcome_reason(doc),
            doc,
            att,
            links,
            leverage,
            stored=doc,
            content=content,
            legacy_unverified=legacy,
        )

    def _check(
        self, date: str, doc: Mapping[str, Any], content: AttemptContent
    ) -> tuple[str, tuple[str, ...], tuple[str, str] | None, bool]:
        """``(mismatch, waits, origin, unsettled)`` of one stored outcome with its verified
        files: ``mismatch`` is ``""`` when its record recomputes equal (:func:`dependency_record`,
        at the record's own version); ``origin`` names the stale dependency a mismatch comes
        from; ``unsettled`` when a dependency could not be read consistently."""
        stored = doc.get("record")
        if not isinstance(stored, Mapping):
            return "stored without a dependency record (an older volsto-backtest)", (), None, False
        version = stored.get("version")
        if version not in RECORD_VERSIONS:
            return (
                f"record version {version!r} (this code reads {RECORD_VERSIONS})",
                (),
                None,
                False,
            )
        status = str(doc.get("status"))
        if status not in (*RESULT_STATUSES, "failed", "skipped"):
            return f"unknown stored status {status!r}", (), None, False
        waits: list[str] = []
        judged: dict[str, Verdict] = {}
        if status != "skipped":
            deps = set(self.dependencies(date)[0]) if status in RESULT_STATUSES else set()
            p = self.previous(date)
            if p is not None:
                deps.add(p)
            for e in sorted(deps):
                v = judged[e] = self.verdict(e)
                if v.status == "missing":
                    waits.append(e)
                elif v.status == "pending":
                    waits += v.waits
                elif v.status == "unsettled":
                    return f"depends on unsettled {e}: {v.reason}", (), (e, v.reason), True
                elif v.status == "stale":
                    origin = v.origin if v.origin is not None else (e, v.reason)
                    return f"depends on stale {origin[0]}: {origin[1]}", (), origin, False
        raw_links, raw_prev = stored.get("links"), stored.get("previous")
        links: Mapping[str, Any] = raw_links if isinstance(raw_links, Mapping) else {}
        prev: Mapping[str, Any] = raw_prev if isinstance(raw_prev, Mapping) else {}

        def link_of(e: str) -> tuple[str | None, bool]:
            v = judged[e]
            if v.status == "missing":  # unconfirmable yet: the stored link stands in (pending)
                lev = bool(prev.get("leverage")) if prev.get("date") == e else False
                return links.get(e), lev
            return v.links.get(int(version), UNAVAILABLE), v.leverage

        expected = dependency_record(
            self, date, doc, content.files, link_of, inputs=self, version=int(version)
        )
        mismatch = _record_mismatch(stored, expected)
        if not mismatch and status in RESULT_STATUSES:
            # the snapshot the record names is intact, but the importer that wrote it is not
            # the current one: a fresh import would differ (SPEC §13.1, 2026-09-22)
            tag = self.read_snapshot(date).importer_tag
            if tag != IMPORTER_TAG:
                mismatch = importer_tag_reason(tag)
        return mismatch, tuple(dict.fromkeys(waits)), None, False

    def _own_links(
        self, date: str, doc: Mapping[str, Any], files: AttemptContent
    ) -> tuple[dict[int, str], bool]:
        """The date's state links (every record version) from the verified bytes — never a new
        read of the attempt's files; a version-1 record has no leverage content, so its
        version-2 link takes the content of the cache entry its key names."""
        record = doc["record"]
        lev = record.get("leverage")
        if not isinstance(lev, Mapping):
            return {}, False
        if doc.get("status") in RESULT_STATUSES:
            params = files.json(FIT_NAME).get("params")
        else:
            params = doc.get("params")
        if params is None:
            return {}, False
        key = str(lev["key"])
        content = lev["content"] if "content" in lev else self.leverage_state(key)[1]
        snap = str(record["snapshot"])
        p = state_params(params)
        links = {v: state_link(date, snap, p, key, content, version=v) for v in RECORD_VERSIONS}
        return links, bool(lev.get("complete"))


def _outcome_reason(doc: Mapping[str, Any]) -> str:
    reasons = doc.get("incomplete_reasons") or [doc.get("error", "")]
    return "; ".join(str(r) for r in reasons if r)


def dependency_record(
    ledger: Ledger,
    date: str,
    doc: Mapping[str, Any],
    files: Mapping[str, bytes],
    link_of: Callable[[str], tuple[str | None, bool]],
    *,
    inputs: RecordInputs,
    version: int = RECORD_VERSION,
) -> dict[str, Any]:
    """**The integrity function**: the dependency record (at record ``version``) of the stored
    outcome ``doc`` of ``date`` — the config hash; the inputs digest (:class:`InputIndex`); the
    calibration code tag; the snapshot digest; for results, the checksum and row count of
    ``rows.parquet`` and the checksum of ``fit.json`` (both from ``files``, the outcome's
    bytes), the leverage key recomputed from the snapshot, the fitted parameters and the base
    spec, its completeness in the configured cache and (version 2) the digest of its numbers,
    the previous calendar date with whether its leverage was used, the state link
    (:func:`state_link`) of every date whose marked state the rows use and the snapshot digest of
    every date whose close they use (:func:`dependencies_of`), the links given by ``link_of`` as
    ``(link, leverage complete)``; for a failure, the previous date and, when the date was
    marked, the leverage of its own parameters; for a skip, its (empty) files and (version 2) its
    cause, re-derived: the day file is absent.

    ``inputs`` says where the snapshot digests and the leverage come from: ``run`` passes what
    it USED (the digests of the bytes it parsed, the leverage it priced with) and stores the
    result in the outcome (``record``); :meth:`Ledger.verdict` passes itself (the world as it is)
    with the dependencies' confirmed links, and the date is confirmed iff the two are equal."""
    status = str(doc.get("status"))
    record: dict[str, Any] = {
        "version": version,
        "date": date,
        "status": status,
        "config_hash": ledger.hash,
        "inputs_digest": ledger.inputs_digest(date),
        "calibration_code_tag": CALIBRATION_CODE_TAG,
        "snapshot": inputs.snapshot(date),
    }
    if status in (*RESULT_STATUSES, "skipped"):
        record["rows"] = rows_fingerprint(files.get(ROWS_NAME))
        record["fit"] = bytes_digest(files.get(FIT_NAME))
    if status == "skipped" and version >= 2:
        try:
            absent = ledger.inputs.file_sha(date) == ABSENT
        except (DateFailure, OSError):
            absent = False
        record["cause"] = SKIP_CAUSE if absent else "the day file exists"
    if status in RESULT_STATUSES:
        params: Any = None
        if FIT_NAME in files:
            try:
                params = json.loads(files[FIT_NAME].decode("utf-8")).get("params")
            except (ValueError, UnicodeDecodeError, AttributeError):
                params = None
        if isinstance(params, Mapping):
            record["leverage"] = inputs.leverage(date, params, version)
        else:
            record["leverage"] = {"key": "unrecoverable: no fitted parameters", "complete": False}
            if version >= 2:
                record["leverage"]["content"] = None
        p = ledger.previous(date)
        record["previous"] = None if p is None else {"date": p, "leverage": link_of(p)[1]}
        record["links"] = {e: link_of(e)[0] for e in ledger.dependencies(date)[0]}
        record["closes"] = {e: inputs.snapshot(e) for e in ledger.dependencies(date)[1]}
    elif status == "failed":
        params = doc.get("params")
        if isinstance(params, Mapping):
            record["leverage"] = inputs.leverage(date, params, version)
        p = ledger.previous(date)
        record["previous"] = None if p is None else {"date": p}
    return record


# --------------------------------------------------------------------------------------------
# writers: the pointer rule, adoption, garbage collection, migration of the flat layout
# --------------------------------------------------------------------------------------------


def header_refusal(cfg: BacktestConfig, store: BacktestStore, *, force: bool) -> str:
    """Why this config may not write into the store (``""`` when it may) — a PURE check: the
    store header names another config hash and ``--force`` was not given."""
    header = store.header()
    h = cfg.content_hash()
    if header is None or header.get("config_hash") == h or force:
        return ""
    return (
        f"the store {store.root} was built from config hash "
        f"{str(header.get('config_hash'))[:12]} and this config hashes to {h[:12]}: a changed "
        "config would mix results. Recompute every date under this config with: "
        f"{backtest_command(cfg, 'run', [], extras=['--force'])} — or write elsewhere (replace "
        f"{NEW_STORE}): {elsewhere_command(cfg, [])}"
    )


def foreign_pointer_refusal(store: BacktestStore) -> str:
    """Why no mutating command may touch the store (``""`` when none): a ``CURRENT`` written
    by another pointer version (a newer volsto-backtest).  A pure check."""
    bad = [d for d in store.dates() if _foreign_version(store.pointer_bytes(d))]
    if not bad:
        return ""
    more = " ..." if len(bad) > 3 else ""
    return (
        f"{len(bad)} date(s) of {store.root} ({', '.join(bad[:3])}{more}) "
        f"have a CURRENT of another pointer version (this code reads version "
        f"{POINTER_VERSION}): a newer volsto-backtest wrote them; use that version"
    )


def bind_header(cfg: BacktestConfig, store: BacktestStore) -> None:
    """Write the header when it is missing or names another config (after every refusal)."""
    header = store.header()
    if header is None or header.get("config_hash") != cfg.content_hash():
        store.write_header(cfg)


def check_store(cfg: BacktestConfig, store: BacktestStore, *, force: bool) -> None:
    """:func:`header_refusal` then :func:`bind_header` (``run`` calls them apart, with its other
    refusals in between)."""
    why = header_refusal(cfg, store, force=force)
    if why:
        raise RefusedError(why)
    bind_header(cfg, store)


def _decide(
    store: BacktestStore, ledger: Ledger, held: DateLock, new: Attempt | None
) -> tuple[Verdict | None, Verdict, Verdict | None]:
    """``(choice, current, new's verdict)`` of the pointer rule for one date, from a fresh read
    under the date's lock (:func:`choose_pointer`).  A ``CURRENT`` of another pointer version is
    refused, never judged."""
    date = held.date
    foreign = _foreign_version(held.read_pointer())
    if foreign:
        raise RefusedError(f"{date}: {foreign}")
    ledger.invalidate(date)
    current = ledger.verdict(date)
    if current.flat:  # not migrated yet: there is no CURRENT to protect
        current = Verdict(date, "missing", current.reason)
    verdicts = ledger.attempt_verdicts(date, current)
    new_v = next(
        (
            v
            for v in verdicts
            if new is not None and v.attempt is not None and v.attempt.id == new.id
        ),
        None,
    )
    choice = choose_pointer(current, verdicts, ledger.hash, new_v)
    ptr = parse_pointer(held.read_pointer())
    named = None if ptr is None or "invalid" in ptr else ptr.get("attempt")
    if (
        choice is not None
        and choice.attempt is not None
        and current.stored is not None
        and choice.attempt.id == named
    ):
        choice = None  # already current (and its pointer verifies)
    return choice, current, new_v


def _point(store: BacktestStore, held: DateLock, choice: Verdict) -> None:
    assert choice.attempt is not None and choice.content is not None
    store.set_pointer(held, choice.attempt, choice.stored_status, choice.content.listing)


def publish_outcome(
    store: BacktestStore, ledger: Ledger, date: str, staged: Path, label: str
) -> tuple[Attempt, Verdict, bool]:
    """Publish a complete staged outcome and apply the pointer rule, all under the date's lock,
    from a fresh consistent read of the store: ``(attempt, its verdict, whether CURRENT names
    it)``."""
    with store.lock(date) as held:
        _crash_point("lock.acquired")
        foreign = _foreign_version(held.read_pointer())
        if foreign:  # refused before anything is published into a date another version owns
            raise RefusedError(f"{date}: {foreign}")
        attempt = store.publish(held, staged, label)
        ledger.invalidate()
        choice, _, new_v = _decide(store, ledger, held, attempt)
        _crash_point("pointer.decided")
        if choice is not None:
            _point(store, held, choice)
        ptr = parse_pointer(held.read_pointer())
        is_current = ptr is not None and ptr.get("attempt") == attempt.id
        if not held.still_at(store.date_dir(date)):
            raise RefusedError(
                f"{date}: the date directory {store.date_dir(date)} was moved or replaced while "
                f"it was locked: the attempt {attempt.id} went into the moved directory, not "
                "into the store (moving a store that a writer uses is not supported)"
            )
    _crash_point("lock.released")
    ledger.invalidate(date)
    if new_v is None:
        new_v = ledger.candidate(date, attempt)
    return attempt, new_v, is_current


def _adopt_held(
    store: BacktestStore, ledger: Ledger, held: Mapping[str, DateLock], dates: Sequence[str]
) -> list[str]:
    adopted: list[str] = []
    ledger.invalidate()
    for d in dates:
        choice, _, _ = _decide(store, ledger, held[d], None)
        if choice is not None:
            _point(store, held[d], choice)
            adopted.append(d)
            log.warning(
                "%s: adopted the published attempt %s (%s)",
                d,
                choice.attempt.id if choice.attempt else "",
                choice.status,
            )
            ledger.invalidate(d)
    return adopted


def adopt(store: BacktestStore, ledger: Ledger, dates: Sequence[str]) -> list[str]:
    """Apply the pointer rule without a new attempt to every date of ``dates`` that has
    published attempts (under their locks, taken together, from one fresh read): ``CURRENT``
    moves to an attempt of this config that protects more — what a writer killed between
    publishing and pointing left, or an older result that verifies again.  Refused (before any
    write) when a ``CURRENT`` of another pointer version exists."""
    why = foreign_pointer_refusal(store)
    if why:
        raise RefusedError(why)
    todo = [d for d in sorted(set(dates)) if store.attempts_dir(d).is_dir()]
    if not todo:
        return []
    with store.locks(todo) as held:
        return _adopt_held(store, ledger, held, todo)


def gc_refusal(store: BacktestStore, ledger: Ledger) -> str:
    """Why ``gc`` may not run (``""`` when it may): a store whose header is not this config's,
    or a ``CURRENT`` of another pointer version.  A pure check."""
    header = store.header()
    if header is None or header.get("config_hash") != ledger.hash:
        return (
            f"gc: the store {store.root} was built from config hash "
            f"{str((header or {}).get('config_hash'))[:12]} and this config hashes to "
            f"{ledger.hash[:12]}: gc only runs with the store's own config"
        )
    return foreign_pointer_refusal(store)


def collect_garbage(store: BacktestStore, ledger: Ledger) -> dict[str, int]:
    """``volsto-backtest gc`` (conservative and bound to the config).  Refused (before any
    write) unless the store header names this config and every ``CURRENT`` is of this pointer
    version.  Under every date's lock (taken together; writers wait), every mutation through the
    locked descriptors: the pointer rule is applied (:func:`adopt`), then — from one fresh read
    — a date whose FULL verdict is ``done`` loses its non-current attempts (each either holds no
    results or is superseded by the current done attempt; symbolic links are skipped with a
    warning); any other date keeps all of its attempts.  Pointer temporaries and the staging
    directories of dead writers on this host go too; those of other hosts are kept and
    counted.  Migrated flat leftovers are removed under the store lock."""
    why = gc_refusal(store, ledger)
    if why:
        raise RefusedError(why)
    counts = {
        "attempts": 0,
        "kept": 0,
        "symlinks": 0,
        "staging": 0,
        "staging_kept": 0,
        "pointer": 0,
        "flat": 0,
    }
    dates = store.dates()
    with store.locks(dates) as held:
        judged = [
            d for d in dates if d in ledger.vendor_calendar and store.attempts_dir(d).is_dir()
        ]
        _adopt_held(store, ledger, held, judged)
        ledger.invalidate()
        for d in dates:
            h = held[d]
            v = ledger.verdict(d) if d in ledger.vendor_calendar else None
            done = (
                v is not None and v.status == "done" and v.attempt is not None and v.results_bearing
            )
            names = h.attempt_names()
            for name in names:
                if name.startswith(STAGING_PREFIX):
                    if _staging_owner_alive(Path(name)):
                        counts["staging_kept"] += 1
                    else:
                        shutil.rmtree(name, dir_fd=h.attempts_fd(), ignore_errors=True)
                        counts["staging"] += 1
                    continue
                if not _ATTEMPT_NAME.fullmatch(name):
                    continue
                if v is not None and v.attempt is not None and name == v.attempt.id:
                    continue
                if not done:
                    counts["kept"] += 1
                    continue
                mode = os.stat(name, dir_fd=h.attempts_fd(), follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    log.warning("gc: %s: %s is a symbolic link: skipped", d, name)
                    counts["symlinks"] += 1
                    continue
                store.remove_attempt(h, store.attempt(d, name))
                counts["attempts"] += 1
            for name in h.names():
                if name.startswith(POINTER_TMP_PREFIX):
                    h.unlink(name)  # writers write pointers under this lock only
                    counts["pointer"] += 1
            h.release_attempts()
        ledger.invalidate()
    with store.store_lock():
        counts["flat"] = _remove_imported_leftovers(store)
    return counts


def import_record_path(snapshot: Path) -> Path:
    """``<snapshots>/<underlying>_<date>.import.json``: what a snapshot was imported from."""
    return snapshot.with_name(snapshot.stem + IMPORT_SUFFIX)


def snapshot_bound(
    cfg: BacktestConfig,
    path: Path,
    source_sha: str,
    manifest_sha: str,
    data: bytes | None = None,
) -> bool:
    """A stored snapshot is reused only when its import record binds its content digest
    (:func:`snapshot_digest`; of ``data``, the bytes the caller read, when given) to the current
    day-file and manifest-entry digests and the configured eSSVI / calendar-repair settings (the
    provenance of the same bytes agreeing as well)."""
    rec = _read_json(import_record_path(path))
    if rec is None:
        return False
    if data is None:
        try:
            data = path.read_bytes()
        except OSError:
            return False
    surf = cfg.section("surface")
    return bool(
        rec.get("snapshot") == snapshot_bytes_digest(data)
        and rec.get("day_file_sha256") == source_sha
        and rec.get("manifest_sha256") == manifest_sha
        and rec.get("essvi") == bool(surf["essvi"])
        and rec.get("calendar_repair") == bool(surf["calendar_repair"])
        and _snapshot_ok(cfg, data, source_sha, manifest_sha)
    )


def bind_snapshot(
    cfg: BacktestConfig,
    path: Path,
    source_sha: str,
    manifest_sha: str,
    how: str,
    digest: str,
) -> None:
    """Write the import record of a snapshot whose content digest is ``digest`` (computed by
    the caller from the bytes it wrote or checked)."""
    surf = cfg.section("surface")
    _write_json(
        import_record_path(path),
        {
            "snapshot": digest,
            "day_file_sha256": source_sha,
            "manifest_sha256": manifest_sha,
            "essvi": bool(surf["essvi"]),
            "calendar_repair": bool(surf["calendar_repair"]),
            "bound_by": how,
            "created_utc": _utc_now(),
            "code_version": code_version(),
        },
    )


def import_snapshot(
    cfg: BacktestConfig, date: str, source_sha: str, manifest_sha: str, dest: Path
) -> str:
    """Import a vendor day into ``dest`` (with the backtest's provenance entry), atomically;
    returns the snapshot digest of the bytes written."""
    surf = cfg.section("surface")
    try:
        cfg_map, _, _, _ = import_day(
            cfg.data_root,
            date,
            cfg.underlying,
            essvi=bool(surf["essvi"]),
            calendar_repair=DEFAULT_CALENDAR_REPAIR if surf["calendar_repair"] else None,
        )
    except Exception as exc:
        raise ImportFailure(f"{date}: import failed: {type(exc).__name__}: {exc}") from exc
    cfg_map["provenance"][SNAPSHOT_INPUTS] = {
        "day_file_sha256": source_sha,
        "manifest_sha256": manifest_sha,
    }
    digest: list[str] = []

    def write(tmp: Path) -> None:
        write_snapshot(cfg_map, tmp)
        digest.append(snapshot_bytes_digest(tmp.read_bytes()))  # the bytes renamed to dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(dest, write)
    return digest[0]


def _stage_copy(store: BacktestStore, date: str, files: Mapping[str, Path]) -> Path:
    staged = store.stage(date)
    for name, src in files.items():
        shutil.copyfile(src, staged / name)
        _fsync_path(staged / name)
    _fsync_path(staged)
    return staged


def _quarantine(store: BacktestStore, held: DateLock, names: Sequence[str]) -> str:
    """Move flat files no attempt holds into ``quarantine/<date>-<digest>/`` (never deleted),
    relative to the locked date directory."""
    listing = {n: _file_sha256(store.date_dir(held.date) / n) for n in names}
    rel = f"{held.date}-{listing_digest(listing)[:16]}"
    dest = store.quarantine_root / rel
    dest.mkdir(parents=True, exist_ok=True)
    qfd = os.open(dest, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        for n in names:
            if _exists_at(n, qfd) and _sha256((dest / n).read_bytes()) == listing[n]:
                held.unlink(n)  # moved there by an interrupted earlier migration
            else:
                os.replace(n, n, src_dir_fd=held.fd, dst_dir_fd=qfd)
        os.fsync(qfd)
    finally:
        os.close(qfd)
    return f"{QUARANTINE_DIR}/{rel}"


def _legacy_label(done: Path) -> str:
    """``legacy-<status>`` of a flat ``done.json`` (``legacy-unknown`` when it is torn or its
    status is not a plain word: the attempt is judged by its content anyway)."""
    status = str((_read_json(done) or {}).get("status", "unknown"))
    return LEGACY_PREFIX + (status if re.fullmatch(r"[a-z][a-z-]*", status) else "unknown")


def _migrate_date(
    store: BacktestStore, ledger: Ledger, held: DateLock, leftovers: Sequence[Path]
) -> dict[str, Any]:
    """One date of :func:`migrate_store`, under its lock, every mutation through ``held``."""
    date = held.date
    d = store.date_dir(date)
    flat = [n for n in LEGACY_FILES if n in held.names() and (d / n).is_file()]
    entry: dict[str, Any] = {
        "flat_files": {n: _file_sha256(d / n) for n in flat},
        "attempts": [],
        "leftovers": [p.name for p in leftovers],
    }
    published: list[Attempt] = []
    if DONE_NAME in flat:
        files = {n: d / n for n in (ROWS_NAME, FIT_NAME, DONE_NAME) if n in flat}
        label = _legacy_label(d / DONE_NAME)
        published.append(store.publish(held, _stage_copy(store, date, files), label))
    if FAILURE_NAME in flat:
        failed = _stage_copy(store, date, {DONE_NAME: d / FAILURE_NAME})
        published.append(store.publish(held, failed, f"{LEGACY_PREFIX}failed"))
    for p in leftovers:  # complete directories of an interrupted flat-layout commit
        try:
            names = {q.name: q for q in p.iterdir() if q.is_file()}
        except FileNotFoundError:
            continue  # handled by another process
        body = {n: names[n] for n in (ROWS_NAME, FIT_NAME, DONE_NAME) if n in names}
        if DONE_NAME in body:
            label = _legacy_label(body[DONE_NAME])
            published.append(store.publish(held, _stage_copy(store, date, body), label))
        if FAILURE_NAME in names:
            staged = _stage_copy(store, date, {DONE_NAME: names[FAILURE_NAME]})
            published.append(store.publish(held, staged, f"{LEGACY_PREFIX}failed"))
    entry["attempts"] = [a.id for a in published]
    _crash_point("migrate.published")
    # always the pointer rule (config, rank and results checks included)
    choice, _, _ = _decide(store, ledger, held, None)
    if choice is not None:
        _point(store, held, choice)
    entry["current"] = (parse_pointer(held.read_pointer()) or {}).get("attempt")
    _crash_point("migrate.pointed")
    kept = {h for a in published for h in a.listing().values()}
    lost = [n for n in flat if _file_sha256(d / n) not in kept]
    for n in flat:
        if n not in lost:  # its bytes are inside a published attempt
            held.unlink(n)
    if lost:
        entry["quarantined"] = _quarantine(store, held, lost)
        log.warning(
            "%s: %d flat file(s) held by no attempt moved to %s",
            date,
            len(lost),
            entry["quarantined"],
        )
    os.fsync(held.fd)
    _crash_point("migrate.flat_removed")
    ledger.invalidate(date)
    return entry


def _remove_imported_leftovers(store: BacktestStore) -> int:
    """Remove the flat-layout leftovers a migration marked imported, then an empty ``.staging``
    (call under :meth:`BacktestStore.store_lock`)."""
    root = store.root / LEGACY_STAGING_DIR
    n = 0
    if not root.is_dir():
        return n
    for p in sorted(root.glob(f"*{IMPORTED_SUFFIX}")):
        shutil.rmtree(p, ignore_errors=True)
        n += 1
    with contextlib.suppress(OSError):
        root.rmdir()  # only when empty (incomplete leftovers stay listed)
    return n


def _flat_leftovers(store: BacktestStore) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    flat_staging = store.root / LEGACY_STAGING_DIR
    try:
        entries = sorted(flat_staging.iterdir())
    except (FileNotFoundError, NotADirectoryError):  # none, or removed by a migration meanwhile
        return out
    for p in entries:
        m = _ISO_PREFIX.match(p.name)
        if (
            p.is_dir()
            and m
            and not p.name.endswith(IMPORTED_SUFFIX)
            and any((p / n).is_file() for n in (DONE_NAME, FAILURE_NAME))
        ):
            out.setdefault(m.group(1), []).append(p)
    return out


def _migration_journal(store: BacktestStore) -> tuple[list[dict[str, Any]], str]:
    """``(history, problem)``: the migration journal, or ``([], why)`` when ``migrations.json``
    exists but cannot be read as one (it is then kept as it is, never overwritten)."""
    path = store.root / MIGRATIONS_NAME
    if not path.exists():
        return [], ""
    data = _read_json(path)
    history = None if data is None else data.get("migrations")
    if not isinstance(history, list) or not all(isinstance(e, dict) for e in history):
        return [], (
            f"{_display(path)} exists but is not a readable migration journal (kept as it is, "
            "never overwritten)"
        )
    return list(history), ""


def _migration_log(store: BacktestStore) -> list[dict[str, Any]]:
    history, problem = _migration_journal(store)
    if problem:
        raise RefusedError(f"migrate: {problem}")
    return history


def _write_migration_log(store: BacktestStore, history: list[dict[str, Any]]) -> None:
    _write_json(store.root / MIGRATIONS_NAME, {"migrations": history})


def _stored_config_hash(store: BacktestStore, date: str) -> str:
    """The config hash a stored date names — its flat ``done.json``, else the ``done.json`` of
    the attempt its ``CURRENT`` names (unverified: a refusal reads it) — or ``""``."""
    d = store.date_dir(date)
    doc = _read_json(d / DONE_NAME)
    if doc is None:
        cur = store.current(date)
        doc = None if cur is None else _read_json(cur.path / DONE_NAME)
    return str((doc or {}).get("config_hash") or "")


def migration_refusal(store: BacktestStore, ledger: Ledger, *, rebinding: bool = False) -> str:
    """Why this config may not migrate the store (``""`` when it may or there is nothing to
    migrate) — a pure check, made by ``run`` with its other refusals before anything is
    written: an unreadable journal (kept); another config's header — or, without a header, a
    stored date computed under another config (every stored date is judged: calendar dates by
    :func:`computed_under_another_config`, the others by the config hash they name); an
    interrupted migration another config started; a CURRENT of another version.  With
    ``rebinding`` (``run --force``, which binds this config's header first) the header and
    headerless checks are what ``--force`` accepts."""
    history, problem = _migration_journal(store)
    flat = any(store.legacy_files(d) for d in store.dates()) or bool(_flat_leftovers(store))
    pending = bool(history) and history[-1].get("state") == "in_progress"
    if problem and flat:
        return (
            f"migrate: {problem}; move it aside (e.g. into {_display(store.quarantine_root)}/) "
            "and rerun: the dates an interrupted migration had moved are then recorded from their "
            "legacy-* attempts"
        )
    if not (flat or pending):
        return ""
    started = str(history[-1].get("config_hash", "")) if pending else ""
    header = store.header()
    owner = str((header or {}).get("config_hash") or "")
    if header is not None and owner != ledger.hash and not rebinding:
        who = started if started and started != owner else owner
        return (
            f"migrate: the store {store.root} was built from config hash {owner[:12]} and this "
            f"config hashes to {ledger.hash[:12]}: migrate it with config hash {who[:12]}, "
            + (
                "which started the interrupted migration"
                if who != owner
                else "the store's own config"
            )
            + " (or rebuild it with run --force)"
        )
    if header is None and not rebinding:
        foreign: dict[str, str] = {}
        for d in store.dates():
            if d in ledger.vendor_calendar:
                v = ledger.verdict(d)
                if computed_under_another_config(v, ledger.hash):
                    foreign[d] = v.stored_hash
            else:
                h = _stored_config_hash(store, d)
                if h and h != ledger.hash:
                    foreign[d] = h
        if foreign:
            dates = sorted(foreign)
            hashes = sorted({h[:12] for h in foreign.values() if h})
            more = " ..." if len(dates) > 4 else ""
            return (
                f"migrate: the store {store.root} has no header and {len(dates)} date(s) "
                f"({', '.join(dates[:4])}{more}) hold results computed under config hash "
                f"{', '.join(hashes) or 'unknown'} (this config hashes to {ledger.hash[:12]}): "
                "migrate it with that config (or rebuild it with run --force)"
            )
    if started and started != ledger.hash:
        return (
            f"migrate: an interrupted migration of {store.root} was started under config hash "
            f"{started[:12]} and this config hashes to {ledger.hash[:12]}: finish it with that "
            "config"
        )
    return foreign_pointer_refusal(store)


def migrate_store(store: BacktestStore, ledger: Ledger) -> dict[str, dict[str, Any]]:
    """Move a store of the flat layout (``dates/<date>/{rows.parquet, fit.json, done.json,
    failure.json}`` and the ``.staging`` directory of its commits) into attempts, in place.

    Refused (:class:`RefusedError`, nothing written) unless the store header names this config
    (or there is none) and no ``CURRENT`` is of another pointer version; a store that is not
    writable raises :class:`ConfigError` before any lock.  The whole migration runs under the
    store lock (a concurrent migration waits, then finds nothing to do), each date under its
    lock, and is JOURNALLED: an ``in_progress`` entry of ``migrations.json`` is written before
    the first destructive step, each date's report right after that date, and the entry is
    completed after the snapshot binding — an interrupted migration is finished by the next
    open (the reports of dates it had already moved are rebuilt from their ``legacy-*``
    attempts).  Per date: each flat outcome is copied into ``attempts/legacy-<status>-<digest>/``
    (content-addressed: re-running finds it published); ``CURRENT`` is set by the pointer rule
    (:func:`choose_pointer`, so an attempt of another config, a torn ``done.json`` or a result
    that ranks below a published one never becomes current); only then are the flat files
    removed — those no attempt holds are moved to ``quarantine/``; leftovers are renamed
    ``*.imported`` before their removal.  The snapshots of the migrated dates whose results
    verify are then bound only if a fresh import of the day file reproduces them (the same
    import within the surface tolerance, :func:`_bind_migrated_snapshots`); otherwise they stay
    unbound (the date is recomputed) and the mismatch is recorded."""
    why = migration_refusal(store, ledger)
    if why:
        raise RefusedError(why)
    todo_hint = sorted(
        {d for d in store.dates() if store.legacy_files(d)} | set(_flat_leftovers(store))
    )
    history, problem = _migration_journal(store)
    if problem:  # nothing flat (else refused above): the unreadable journal is left alone
        return {}
    resuming = bool(history) and history[-1].get("state") == "in_progress"
    if not todo_hint and not resuming:
        return {}
    check_writable(store, ledger, todo_hint)
    report: dict[str, dict[str, Any]] = {}
    with store.store_lock():
        leftovers = _flat_leftovers(store)
        todo = sorted({d for d in store.dates() if store.legacy_files(d)} | set(leftovers))
        history = _migration_log(store)
        if history and history[-1].get("state") == "in_progress":
            entry = history[-1]
        else:
            if not todo:
                return {}
            if not history:  # no journal: dates an unrecorded migration moved are recorded too
                todo = sorted(
                    set(todo)
                    | {
                        d
                        for d in store.dates()
                        if any(a.id.startswith(LEGACY_PREFIX) for a in store.attempts(d))
                    }
                )
            entry = {
                "state": "in_progress",
                "config_hash": ledger.hash,
                "from": "flat layout (dates/<date>/{rows.parquet, fit.json, done.json})",
                "to": f"attempts + {CURRENT_NAME} (pointer version {POINTER_VERSION})",
                "todo": todo,
                "dates": {},
                "started_utc": _utc_now(),
                "code_version": code_version(),
                "volsto_version": volsto_version(),
                "host": socket.gethostname(),
            }
            history.append(entry)
        entry["todo"] = sorted(set(entry.get("todo", [])) | set(todo))
        _write_migration_log(store, history)
        _crash_point("migrate.journalled")
        for d in todo:
            if not store.legacy_files(d) and d not in leftovers:
                continue  # moved before (its report is rebuilt below)
            with store.lock(d) as held:
                report[d] = _migrate_date(store, ledger, held, leftovers.get(d, []))
            entry["dates"][d] = report[d]
            _write_migration_log(store, history)
            _crash_point("migrate.reported")
        for d in entry["todo"]:  # moved by an interrupted run, its report lost: rebuild it
            if d not in entry["dates"]:
                entry["dates"][d] = {
                    "attempts": [a.id for a in store.attempts(d) if a.id.startswith(LEGACY_PREFIX)],
                    "current": (store.pointer(d) or {}).get("attempt"),
                    "rebuilt": True,
                }
        for paths in leftovers.values():
            for p in paths:
                with contextlib.suppress(FileNotFoundError):
                    os.replace(p, p.with_name(p.name + IMPORTED_SUFFIX))
                _crash_point("migrate.leftover_marked")
        _remove_imported_leftovers(store)
        _crash_point("migrate.leftover_removed")
        ledger.invalidate()
        bound, unbound = _bind_migrated_snapshots(ledger, sorted(entry["dates"]))
        entry["snapshots_bound"] = sorted(set(entry.get("snapshots_bound", [])) | set(bound))
        entry["snapshots_unbound"] = unbound
        _crash_point("migrate.bound")
        entry["state"] = "complete"
        entry["utc"] = _utc_now()
        _write_migration_log(store, history)
    log.warning(
        "store %s: migrated %d date(s) from the flat layout; bound %d snapshot(s), %d unbound",
        _display(store.root),
        len(entry["dates"]),
        len(bound),
        len(unbound),
    )
    return {**entry["dates"], **report}


def check_writable(
    store: BacktestStore,
    ledger: Ledger,
    dates: Sequence[str] | None = None,
    *,
    attempts: bool = False,
) -> None:
    """A store a mutating command must write (the flat layout to migrate included; with
    ``attempts``, every date's ``attempts/`` too) but cannot: a clear :class:`ConfigError`,
    before any lock or write."""
    if dates is None:
        dates = [d for d in store.dates() if store.legacy_files(d)]
    paths = [store.root, store.dates_root, *(store.date_dir(d) for d in dates)]
    if attempts:
        paths += [store.attempts_dir(d) for d in dates]
    paths += [store.root / LEGACY_STAGING_DIR, ledger.snapshots]
    bad = [p for p in paths if p.exists() and not os.access(p, os.W_OK | os.X_OK)]
    if bad:
        flat = (
            " is in the flat layout and must be migrated, but"
            if any(store.legacy_files(d) for d in dates)
            else ":"
        )
        raise ConfigError(
            f"the store {store.root}{flat} {_display(bad[0])} is not writable: make it writable "
            "(chmod -R u+w) or work on a copy"
        )


def _bind_migrated_snapshots(
    ledger: Ledger, dates: Sequence[str]
) -> tuple[list[str], dict[str, str]]:
    """Bind the snapshot of each migrated date whose results verify, when a fresh import of its
    day file reproduces the snapshot — equal in everything but floats and the same surface
    within :data:`volsto.market.compare.SURFACE_TOL_VP` vol points
    (:func:`~volsto.market.compare.snapshot_difference`; until 2026-10-03 the content digests
    had to be equal, which only the machine that wrote the store could meet) — about 1 s per
    date; dates already bound are skipped, so an interrupted binding resumes."""
    bound: list[str] = []
    unbound: dict[str, str] = {}
    for d in dates:
        if d not in ledger.vendor_calendar:
            continue
        v = ledger.verdict(d)
        if not (v.has_results or (v.status == "pending" and v.results_bearing)):
            continue
        snap = ledger.snapshot_path(d)
        rec = _read_json(import_record_path(snap))
        if rec is not None:  # bound already (by an interrupted run of this migration, say)
            if str(rec.get("bound_by", "")).startswith("migration"):
                bound.append(d)
            continue
        if v.doc is None:
            continue
        try:
            source, manifest = ledger.inputs.file_sha(d), ledger.inputs.manifest_sha(d)
        except (DateFailure, OSError) as exc:
            unbound[d] = f"inputs unavailable: {exc}"
            continue
        read, data = read_snapshot(ledger.base, snap)
        digest = read.digest
        if v.doc["record"].get("snapshot") != digest or not _snapshot_ok(
            ledger.cfg, data or b"", source, manifest
        ):
            unbound[d] = "the snapshot is not the one the results were computed from"
            continue
        with tempfile.TemporaryDirectory(prefix="volsto-bind-") as tmp:
            try:
                import_snapshot(ledger.cfg, d, source, manifest, Path(tmp) / snap.name)
            except DateFailure as exc:
                unbound[d] = str(exc)
                continue
            # not by digest: a fresh import on another machine differs in the last bits of
            # every fitted number (CONTRIBUTING.md, "Machine-dependent arithmetic")
            why = snapshot_difference(snap, Path(tmp) / snap.name)
        if why is not None:
            unbound[d] = f"a fresh import of its day file gives another snapshot ({why})"
            continue
        bind_snapshot(
            ledger.cfg, snap, source, manifest, "migration (fresh import reproduced)", digest
        )
        bound.append(d)
    for d, why in unbound.items():
        log.warning("%s: snapshot left unbound (%s): the date will be recomputed if needed", d, why)
    return bound, unbound


@functools.lru_cache(maxsize=1)
def volsto_version() -> dict[str, Any]:
    """The code that wrote an attempt — package version, git commit and dirty flag —
    INFORMATIONAL: stored in ``done.json``, never part of a verdict."""
    import volsto

    commit, dirty = git_state()
    return {"package": volsto.__version__, "git_commit": commit, "git_dirty": dirty}


# --------------------------------------------------------------------------------------------
# per-date states
# --------------------------------------------------------------------------------------------


class DateFailure(RuntimeError):  # noqa: N818 — a date's outcome, not a bug
    """A date that cannot be marked (import, infeasible fit, missing leverage)."""


class ImportFailure(DateFailure):
    """The import of an existing day file raised: a date failure (retried by ``--resume``),
    never a calendar gap."""


class DayFileAbsent(DateFailure):
    """The vendor manifest lists the date as a missing trading day and its file is absent: the
    one deterministic cause ``data.missing_close: skip_date`` drops a date for
    (:data:`SKIP_CAUSE`)."""


class LeverageMissing(DateFailure):
    """A leverage absent from the cache under ``--no-calibrate``."""


class CloseUnavailable(DateFailure):
    """A close of a trade's history is unavailable (``data.missing_close: fail``)."""


@dataclass
class DateState:
    """A marked date: its snapshot, fitted spec, surface, fit and the history quantities."""

    date: str
    snapshot: Path
    source_sha256: str
    spec: CalibrationSpec
    surface: ImpliedSurface
    forward_curve: ForwardCurve
    fit: FitResult
    record: dict[str, Any]
    timings: dict[str, float]
    #: :func:`snapshot_digest` and SHA-256 of the snapshot bytes the state was parsed from
    snapshot_digest: str = ""
    snapshot_sha256: str = ""

    @property
    def state(self) -> RiskState:
        return RiskState(self.spec, None, self.date)

    @property
    def spot(self) -> float:
        """The pricing level: the snapshot's spot (the level the option quotes imply)."""
        return float(self.spec.market.spot)

    @property
    def close(self) -> float:
        """The official close of the date (``market.close``): what trades fix on and are struck
        at.  Snapshots imported before 2026-09-22 carry none and are refused."""
        if self.spec.market.close is None:
            raise DateFailure(
                f"{self.date}: the snapshot {self.snapshot} has no market.close (imported "
                "before 2026-09-22, when spot became the option-implied level): re-import it"
            )
        return float(self.spec.market.close)

    @property
    def discount(self) -> DiscountCurve:
        return self.forward_curve.rate_curve

    @property
    def key(self) -> str:
        return spec_key(self.spec)


#: The provenance entry the backtest adds to every snapshot it imports.
SNAPSHOT_INPUTS = "backtest_inputs"


def _snapshot_ok(
    cfg: BacktestConfig, source: Path | bytes, source_sha: str, manifest_sha: str
) -> bool:
    """A stored snapshot (its bytes, or read from its path) is reused when the backtest imported
    it from the same day file and the same manifest entry (:meth:`InputIndex.manifest_sha`: its
    rates), with the configured eSSVI and calendar-repair settings."""
    try:
        raw = source if isinstance(source, bytes) else source.read_bytes()
        data = yaml.safe_load(raw)
        prov = data["provenance"]
        fit = prov["fit"]
        ins = prov[SNAPSHOT_INPUTS]
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return False
    surf = cfg.section("surface")
    repaired = any(str(k).startswith("calendar_") for k in fit)
    return bool(
        prov.get("file_sha256") == source_sha
        and isinstance(ins, Mapping)
        and ins.get("day_file_sha256") == source_sha
        and ins.get("manifest_sha256") == manifest_sha
        and bool(fit.get("essvi")) == bool(surf["essvi"])
        and repaired == bool(surf["calendar_repair"])
        and prov.get("importer_tag") == IMPORTER_TAG
    )


def fit_record(r: FitResult) -> dict[str, Any]:
    """The JSON record of a marking fit (parameters, status, messages, the per-pillar SSR)."""
    p, b, f, s = r.params, r.breakeven, r.first, r.second
    se = {
        "k1": f.k1_se,
        "lambda1": f.lambda1_se,
        "lambda2": f.lambda2_se,
        "omega1": s.stderr.get("omega1", float("nan")),
        "omega2": s.stderr.get("omega2", float("nan")),
        "chi": s.stderr.get("chi", float("nan")),
    }
    tbl = r.table
    return {
        "status": r.status,
        "messages": list(r.messages),
        "notes": [*r.notes, *f.notes, *s.notes],
        "params": {k: float(getattr(p, k)) for k in PARAMS},
        "breakeven": {k: float(getattr(b, k)) for k in PARAM_COLUMNS},
        "se": se,
        "pillars": [float(x) for x in tbl["T"]],
        "ssr_first_order": [float(x) for x in tbl["ssr_first_order"]],
        "ssr_target": [float(x) for x in r.targets.ssr_target],
        "skew_market": [float(x) for x in tbl["skew_market"]],
        "skew_naked": [float(x) for x in tbl["skew_naked"]],
        "active": list(f.active),
        "bound_flags": list(s.bound_flags),
        "k1_at_bound": bool(f.k1_at_bound),
        "first_objective": float(f.objective),
        "second_objective": float(s.objective),
        "wall_seconds": float(r.wall_seconds),
    }


def pillar_quantities(surface: ImpliedSurface) -> dict[str, list[float]]:
    """VS vol, ATMF vol and ATMF skew at the history pillars (``SurfaceHistory`` sources)."""
    src = SurfacePillarSource(surface)
    ps = [float(t) for t in DEFAULT_PILLARS]
    return {
        "T": ps,
        "vs_vol": [src.vs_vol(t) for t in ps],
        "atm_vol": [src.atm_vol(t) for t in ps],
        "skew": [src.atm_skew(t) for t in ps],
    }


# --------------------------------------------------------------------------------------------
# the run (stage 1)
# --------------------------------------------------------------------------------------------


@dataclass
class DateOutcome:
    date: str
    status: str
    wall_s: float
    calibrated: list[str] = field(default_factory=list)
    n_rows: int = 0
    error: str = ""
    leverage_missing: bool = False
    current: bool = True
    """Whether the date's ``CURRENT`` names this outcome (the pointer rule may keep a better
    one)."""
    attempt: str = ""


#: What the run records for an input it read twice with different content (never equal to a
#: digest: the outcome is judged stale and recomputed).
CHANGED_WHILE_USED = "changed while in use"


def _note_use(used: dict[str, str | None], key: str, digest: str | None) -> None:
    """Record the digest of what was used under ``key``; a second, different one marks the use
    :data:`CHANGED_WHILE_USED`."""
    if used.get(key) is None:
        used[key] = digest
    elif used[key] != digest:
        used[key] = CHANGED_WHILE_USED


class _RecordingCache(LeverageCache):
    """The configured leverage cache, recording the numeric content digest
    (:func:`leverage_content_digest`) of every leverage it loads, from the very bytes the model
    is built from (read once)."""

    def __init__(self, root: str | Path, used: dict[str, str | None]) -> None:
        super().__init__(root)
        self.used = used

    def load(self, spec: CalibrationSpec) -> LeverageFunction:
        key = self.key(spec)
        try:
            data = (self.entry_dir(spec) / LEVERAGE_NAME).read_bytes()
        except FileNotFoundError:
            raise CacheMissError(key) from None
        _note_use(self.used, key, leverage_content_digest(io.BytesIO(data)))
        return LeverageFunction.load(io.BytesIO(data))  # type: ignore[arg-type]


class _UsedInputs:
    """:class:`RecordInputs` of a run: the digests of the snapshot bytes it parsed and the
    leverage it priced with (what :func:`dependency_record` records at commit — nothing is read
    again then)."""

    def __init__(self, run: BacktestRun) -> None:
        self.run = run

    def snapshot(self, date: str) -> str:
        used = self.run.snapshots_used.get(date)
        return self.run.ledger.snapshot(date) if used is None else used

    def leverage(self, date: str, params: Any, version: int) -> dict[str, Any]:
        st = self.run.marked(date)
        if st is None or state_params(params) != state_params(st.fit.params):
            return self.run.ledger.leverage(date, params, version)
        out: dict[str, Any] = {"key": st.key, "complete": self.run.cache.has_key(st.key)}
        if version >= 2:
            out["content"] = self.run.leverage_content(st.key)
        return out


class BacktestRun:
    """Stage 1 (module docstring): marks, prices and attributes dates of one config."""

    def __init__(
        self,
        cfg: BacktestConfig,
        *,
        allow_calibrate: bool,
        snapshots: Path | None = None,
        migrate: bool = False,
    ) -> None:
        """``migrate``: move a store of the flat layout into attempts first
        (:func:`migrate_store`; the stage-1 commands ``run``, ``status``, ``migrate``, ``gc``)."""
        self.cfg = cfg
        self.allow_calibrate = bool(allow_calibrate)
        self.vendor_calendar = calendar(cfg)
        if not self.vendor_calendar:
            raise ConfigError(
                f"no {cfg.underlying} day files in {cfg.data_root} between {cfg.start} and "
                f"{cfg.end}"
            )
        self.store = BacktestStore(cfg.path("out"))
        #: what this process used: snapshot digests by date, leverage content digests by key
        self.snapshots_used: dict[str, str | None] = {}
        self.leverage_used: dict[str, str | None] = {}
        self.cache: LeverageCache = _RecordingCache(cfg.path("cache"), self.leverage_used)
        self.snapshots = snapshots if snapshots is not None else cfg.path("snapshots")
        self.base = cfg.base_spec()
        self.sim = cfg.sim(self.base)
        self.hash = cfg.content_hash()
        self.inputs = InputIndex(cfg, self.vendor_calendar)
        self.ledger = Ledger(
            cfg, self.store, self.inputs, snapshots=self.snapshots, cache=self.cache
        )
        self.migration: dict[str, dict[str, Any]] = {}
        if migrate:
            self.migration = migrate_store(self.store, self.ledger)
            self.ledger.invalidate()
        #: ``skip_date``: dates dropped from the calendar (reason); ``fail``: dates without a
        #: close (reason)
        self.skipped: dict[str, str] = self.ledger.skips()
        self.unavailable: dict[str, str] = {}
        self._states: dict[str, DateState] = {}
        self._failures: dict[str, str] = {}
        self._closes: dict[str, float] = {}
        self._built: dict[str, BuiltTrade] = {}
        self._set_calendar()

    def migrate(self) -> dict[str, dict[str, Any]]:
        """:func:`migrate_store` (after the refusals of ``run``); after a migration the skips and
        the calendar are read again (skips found but not yet written are then found again by
        :meth:`new_skips`)."""
        self.migration = migrate_store(self.store, self.ledger)
        if self.migration:
            self.ledger.invalidate()
            self.skipped = self.ledger.skips()
            self._set_calendar()
        return self.migration

    def _set_calendar(self) -> None:
        self.calendar = [d for d in self.vendor_calendar if d not in self.skipped]
        if not self.calendar:
            raise ConfigError(f"every date of {self.cfg.name} was skipped: {self.skipped}")
        self.index = {d: i for i, d in enumerate(self.calendar)}
        self.trades = book_trades(self.cfg, self.calendar)
        self._built.clear()

    # -- snapshots and closes --------------------------------------------------------------

    def snapshot_path(self, date: str) -> Path:
        return self.snapshots / f"{self.cfg.underlying.lower()}_{date}.yaml"

    def source_sha(self, date: str) -> str:
        sha = self.inputs.file_sha(date)
        if sha == ABSENT:
            raise DayFileAbsent(
                f"{date}: no vendor file {_display(source_file(self.cfg, date))} (the vendor "
                "manifest lists the date as a missing trading day)"
            )
        return sha

    def ensure_snapshot(self, date: str) -> tuple[SnapshotRead, bytes, bool, float]:
        """The date's snapshot, read ONCE — imported first when absent or not bound to its
        sources: ``(the parse of the bytes, the bytes, imported here, seconds)``; the binding
        is checked on the same bytes."""
        path = self.snapshot_path(date)
        sha = self.source_sha(date)
        manifest_sha = self.inputs.manifest_sha(date)
        read, data = read_snapshot(self.base, path)
        if data is not None and snapshot_bound(self.cfg, path, sha, manifest_sha, data):
            if read.spec is None:
                raise DateFailure(f"{date}: unreadable snapshot {_display(path)}: {read.error}")
            return read, data, False, 0.0
        if data is not None:
            log.info("%s: the snapshot is not bound to its sources: re-importing it", date)
        t0 = time.perf_counter()
        digest = import_snapshot(self.cfg, date, sha, manifest_sha, path)
        bind_snapshot(self.cfg, path, sha, manifest_sha, "import", digest)
        self._closes.pop(date, None)
        self.ledger.invalidate()
        read, data = read_snapshot(self.base, path)
        if data is None or read.digest != digest or read.spec is None:
            raise ImportFailure(
                f"{date}: the snapshot {_display(path)} changed while it was imported "
                f"({read.error or 'another content'}): rerun with --resume"
            )
        seconds = time.perf_counter() - t0
        log.info("%s: imported the snapshot in %.1f s -> %s", date, seconds, _display(path))
        return read, data, True, seconds

    def close(self, date: str) -> float:
        """The realised close of a date: ``market.close`` of its snapshot, as parsed (the digest
        of the parsed bytes is what the records name) — the official print, not the
        option-implied ``spot`` (SPEC §13.1)."""
        if date in self.unavailable:
            raise CloseUnavailable(self.unavailable[date])
        if date not in self._closes:
            read, _, _, _ = self.ensure_snapshot(date)
            assert read.spec is not None
            _note_use(self.snapshots_used, date, read.digest)
            if read.spec.market.close is None:
                raise DateFailure(
                    f"{date}: the snapshot has no market.close (imported before 2026-09-22): "
                    "re-import it"
                )
            self._closes[date] = float(read.spec.market.close)
        return self._closes[date]

    def missing_close_message(self, date: str, reason: str, *, absent: bool) -> str:
        src = _display(source_file(self.cfg, date))
        if not absent:  # the day file exists: a failure, never a calendar gap
            return (
                f"the close of {date} is unavailable ({reason}); the day file {src} exists, so "
                "the date is failed, not skipped: fix the file (or the importer) and rerun with "
                "--resume"
            )
        return (
            f"the close of {date} is unavailable ({reason}); fix: restore {src}, or set "
            f"data.missing_close: skip_date to drop {date} from the trading calendar (every "
            "later fixing of a trade spanning it then moves one vendor date)"
        )

    def absent(self, date: str) -> bool:
        """The deterministic cause of a skip: the manifest lists ``date`` and its day file is
        absent (:data:`SKIP_CAUSE`)."""
        try:
            return self.inputs.file_sha(date) == ABSENT
        except (DateFailure, OSError):
            return False

    def new_skips(self, last: str) -> list[str]:
        """``skip_date``: the vendor dates up to ``last`` whose day file is absent and that are
        not skipped yet — dropped from the calendar here (to be committed as ``skipped``).  A
        pure check of the inputs: nothing is imported or written."""
        if self.cfg.missing_close != "skip_date":
            return []
        newly = [
            d
            for d in self.vendor_calendar
            if d <= last and d not in self.skipped and self.absent(d)
        ]
        for d in newly:
            self.skipped[d] = self.missing_close_message(
                d, f"{SKIP_CAUSE}: {_display(source_file(self.cfg, d))}", absent=True
            )
            log.warning("history: %s: dropped from the calendar (skip_date: %s)", d, SKIP_CAUSE)
        if newly:
            self._set_calendar()
        return newly

    def resolve_closes(self, last: str) -> None:
        """Import every vendor date up to ``last`` (the realised histories need their closes).
        A date whose close cannot be had is recorded as unavailable (its own date fails, a trade
        whose history spans it is unpriced) — under ``skip_date`` too: only an absent day file
        is a gap (:meth:`new_skips`), an import that raises is a failure retried by
        ``--resume``."""
        for d in self.vendor_calendar:
            if d > last:
                break
            if d in self.skipped:
                continue
            try:
                self.close(d)
            except CloseUnavailable:
                continue
            except DateFailure as exc:
                self.unavailable[d] = self.missing_close_message(d, str(exc), absent=self.absent(d))
                log.error("history: %s", self.unavailable[d])

    def history(self, inception: str, as_of: str) -> RealisedHistory:
        """The realised closes from ``inception`` to ``as_of`` on the (effective) calendar."""
        i0, i1 = self.index[inception], self.index[as_of]
        dates = self.calendar[i0 : i1 + 1]
        return RealisedHistory(
            dt.date.fromisoformat(inception),
            tuple(dt.date.fromisoformat(d) for d in dates),
            tuple(self.close(d) for d in dates),
        )

    def gaps(self, inception: str, as_of: str) -> list[str]:
        """Skipped vendor dates inside ``(inception, as_of)``."""
        return sorted(d for d in self.skipped if inception < d < as_of)

    # -- states --------------------------------------------------------------------------------

    def state(self, date: str) -> DateState:
        """The marked state of a date (memoised; failures are remembered)."""
        if date in self._states:
            return self._states[date]
        if date in self.unavailable:
            raise CloseUnavailable(self.unavailable[date])
        if date in self._failures:
            raise DateFailure(self._failures[date])
        try:
            st = self._mark(date)
        except DateFailure as exc:
            self._failures[date] = str(exc)
            raise
        except Exception as exc:  # a fit or surface error marks the date as failed
            msg = f"{date}: marking failed: {type(exc).__name__}: {exc}"
            self._failures[date] = msg
            raise DateFailure(msg) from exc
        self._states[date] = st
        return st

    def marked(self, date: str) -> DateState | None:
        """The date's state if this process marked it (never marks)."""
        return self._states.get(date)

    def _mark(self, date: str) -> DateState:
        read, data, _, import_s = self.ensure_snapshot(date)
        path = self.snapshot_path(date)
        _note_use(self.snapshots_used, date, read.digest)
        t0 = time.perf_counter()
        assert read.spec is not None
        spec = read.spec
        fc = ForwardCurve.from_config(spec.market)
        surface = surface_from_config(spec.surface, fc, fc.rate_curve)
        m = self.cfg.section("marking")
        cfg = self.cfg.fit_config()
        parsed = yaml.safe_load(data)
        step0 = None
        if cfg.step0 is not None:
            # the date's SABRW fits, from the bytes just verified (never a second read)
            fits = sabrw_fits_from_config(parsed)
            if fits is None:
                raise DateFailure(f"{date}: the snapshot has no sabrw section: re-import it")
            step0 = step0_source(fits, surface)
        fit = fit_2f_marking(surface, cfg, ssr_target=float(m["ssr_target"]), step0=step0)
        fit_s = time.perf_counter() - t0
        if fit.status == "infeasible":
            raise DateFailure(f"{date}: the marking fit is infeasible: {'; '.join(fit.messages)}")
        spec = dataclasses.replace(spec, model=fit.params)
        t1 = time.perf_counter()
        record = fit_record(fit)
        record["history"] = pillar_quantities(surface)
        if spec.market.close is None:
            raise DateFailure(f"{date}: the snapshot has no market.close: re-import it")
        record["history"]["ln_spot"] = math.log(float(spec.market.close))
        prov = parsed["provenance"]
        record["calendar"] = {
            k: v for k, v in dict(prov.get("fit", {})).items() if str(k).startswith("calendar_")
        }
        record["surface_type"] = type(surface).__name__
        hist_s = time.perf_counter() - t1
        return DateState(
            date,
            path,
            self.source_sha(date),
            spec,
            surface,
            fc,
            fit,
            record,
            {"import": import_s, "fit": fit_s, "history": hist_s},
            snapshot_digest=read.digest,
            snapshot_sha256=_sha256(data),
        )

    def leverage_content(self, key: str) -> str | None:
        """The numeric content digest of the leverage ``key`` as this process uses it: the one it
        loaded for pricing, else the cache entry's when first consumed (memoised)."""
        if self.leverage_used.get(key) is None:
            _note_use(self.leverage_used, key, self.ledger.leverage_state(key)[1])
        return self.leverage_used.get(key)

    def ensure_leverage(self, st: DateState) -> tuple[bool, float]:
        """``(calibrated here, seconds)`` for the date's leverage (its content digest noted as
        used: a different one loaded for pricing later marks the use changed)."""
        if self.cache.has(st.spec):
            self.leverage_content(st.key)
            return False, 0.0
        if not self.allow_calibrate:
            raise LeverageMissing(
                f"{st.date}: leverage {st.key[:16]} is not in the cache {self.cache.root} "
                f"(--no-calibrate); produce it with: {self.command([st.date])}"
            )
        t0 = time.perf_counter()
        log.info(
            "%s: leverage cache miss %s: calibrating at %d particles, horizon %gy",
            st.date,
            st.key[:12],
            st.spec.particle.n_particles,
            st.spec.particle.horizon,
        )
        self.cache.get_or_calibrate(st.spec, allow_calibrate=True)
        self.leverage_content(st.key)
        seconds = time.perf_counter() - t0
        log.info("%s: calibrated in %.1f s", st.date, seconds)
        return True, seconds

    def command(self, dates: Sequence[str]) -> str:
        """The ``volsto-backtest run`` line producing ``dates`` (:func:`backtest_argv`)."""
        return backtest_command(self.cfg, "run", dates)

    # -- trades --------------------------------------------------------------------------------

    def built(self, trade: TradeInstance) -> BuiltTrade:
        if trade.trade_id not in self._built:
            st = self.state(trade.inception)
            # struck at the official close (the level the fixings are observed against)
            self._built[trade.trade_id] = build_product(
                trade.spec, st.close, st.surface, st.discount
            )
        return self._built[trade.trade_id]

    def live(self, date: str) -> list[TradeInstance]:
        """:func:`live_trades` on the run's calendar."""
        return live_trades(self.cfg, self.calendar, self.trades, date)

    def link(self, date: str) -> tuple[str | None, bool]:
        """``(state link, leverage complete)`` of a dependency as this process uses it
        (:func:`state_link`; :data:`UNAVAILABLE` when its state cannot be marked)."""
        try:
            st = self.state(date)
        except DateFailure:
            return UNAVAILABLE, False
        snapshot = self.snapshots_used.get(date) or st.snapshot_digest
        link = state_link(
            date, snapshot, state_params(st.fit.params), st.key, self.leverage_content(st.key)
        )
        return link, self.cache.has_key(st.key)

    def _commit(
        self,
        date: str,
        rows: pd.DataFrame | None,
        fit: Mapping[str, Any] | None,
        done: dict[str, Any],
        *,
        used: DateState | None = None,
        prev_used: bool = False,
    ) -> tuple[Attempt, Verdict, bool]:
        """Write an outcome into a staging directory with its :func:`dependency_record`, publish
        it as an immutable attempt and apply the pointer rule (:func:`publish_outcome`);
        ``(attempt, its verdict, whether it is current)``.  With ``used`` (the date's marked
        state) the record must name its leverage, complete, before anything is published;
        ``prev_used`` says whether the previous date's leverage priced the date's P&L."""
        previous = self.calendar[self.index[date] - 1] if self.index.get(date, 0) > 0 else None

        def link_of(e: str) -> tuple[str | None, bool]:
            link, leverage = self.link(e)
            return link, (leverage and prev_used) if e == previous else leverage

        staged = self.store.stage(date)
        try:
            _crash_point("stage.created")
            if rows is not None:
                rows.reindex(columns=list(ROW_COLUMNS)).to_parquet(staged / ROWS_NAME, index=False)
            _crash_point("stage.rows")
            if fit is not None:
                _write_json(staged / FIT_NAME, fit)
            _crash_point("stage.fit")
            # the record: the staged bytes as written (read once) and what the date USED
            record = dependency_record(
                self.ledger, date, done, read_files(staged), link_of, inputs=_UsedInputs(self)
            )
            _crash_point("stage.record")
            lev = record.get("leverage") or {}
            if used is not None and (
                lev.get("key") != used.key
                or lev.get("complete") is not True
                or lev.get("content") in (None, CHANGED_WHILE_USED)
            ):
                raise DateFailure(
                    f"{date}: the leverage {used.key[:16]} the date used is not complete in the "
                    f"cache, or changed while it was used ({lev}): rerun with --resume"
                )
            if record.get("status") != "skipped" and self.ledger.calendar() != self.calendar:
                raise RuntimeError(
                    "the stored skips do not match this run's calendar (commit the skips first)"
                )
            done["record"] = record
            done.setdefault("volsto_version", volsto_version())  # informational only
            _write_json(staged / DONE_NAME, done)
            _crash_point("stage.done")
            for f in staged.iterdir():
                _fsync_path(f)
            _fsync_path(staged)
            _crash_point("stage.synced")
            return publish_outcome(self.store, self.ledger, date, staged, str(done["status"]))
        except (Exception, KeyboardInterrupt):
            with contextlib.suppress(OSError):  # the store may be unreadable by then
                if staged.is_dir():  # not published: nobody can see it
                    shutil.rmtree(staged, ignore_errors=True)
            raise
        finally:
            self.ledger.invalidate(date)

    def write_skips(self, dates: Sequence[str]) -> None:
        for d in dates:
            done: dict[str, Any] = {
                "date": d,
                "status": "skipped",
                "error": self.skipped[d],
                "config_hash": self.hash,
                "created_utc": _utc_now(),
                "code_version": code_version(),
            }
            empty = pd.DataFrame(columns=list(ROW_COLUMNS))
            _, _, current = self._commit(d, empty, {"date": d, "error": self.skipped[d]}, done)
            if not current:
                log.warning("%s: the skip was recorded but the current attempt protects more", d)
        if dates:
            self.ledger.invalidate()

    # -- the job -------------------------------------------------------------------------------

    def run_date(self, date: str) -> DateOutcome:
        t_start = time.perf_counter()
        i = self.index[date]
        calibrated: list[str] = []
        timings: dict[str, float] = {}
        reasons: list[str] = []
        leverage_missing = False
        try:
            st = self.state(date)
            done_cal, cal_s = self.ensure_leverage(st)
            if done_cal:
                calibrated.append(date)
            timings.update(st.timings)
            timings["calibration"] = cal_s
            prev: DateState | None = None
            prev_note = "" if i > 0 else "first date of the range"
            if i > 0:
                d_prev = self.calendar[i - 1]
                try:
                    prev = self.state(d_prev)
                    done_prev, prev_s = self.ensure_leverage(prev)
                    if done_prev:
                        calibrated.append(d_prev)
                    timings["calibration_previous"] = prev_s
                except DateFailure as exc:
                    prev = None
                    leverage_missing = isinstance(exc, LeverageMissing)
                    prev_note = f"no P&L: the previous date {d_prev} is unavailable: {exc}"
                    log.warning("%s: %s", date, prev_note)
            # the cumulative P&L prices the fresh attributed trades at their inception states
            if prev is not None:
                for d_inc in sorted(
                    {
                        t.inception
                        for t in self.live(date)
                        if attributed(self.cfg, t) and self.index[t.inception] < i - 1
                    }
                ):
                    try:
                        done_inc, inc_s = self.ensure_leverage(self.state(d_inc))
                    except DateFailure as exc:
                        leverage_missing = leverage_missing or isinstance(exc, LeverageMissing)
                        reasons.append(f"no cumulative P&L of the trades struck on {d_inc}: {exc}")
                        log.warning("%s: %s", date, reasons[-1])
                        continue
                    if done_inc:
                        calibrated.append(d_inc)
                    timings["calibration_inception"] = (
                        timings.get("calibration_inception", 0.0) + inc_s
                    )
            base = prev if prev is not None else st
            builder = LSVBuilder(self.cache, base.state, allow_calibrate=False)
            engine = RiskEngine(
                builder, self.sim, max_halvings=int(self.cfg.attribution["max_halvings"])
            )
            t_price = time.perf_counter()
            rows: list[dict[str, Any]] = []
            for trade in self.live(date):
                row = self.trade_row(trade, date, st, prev, engine, prev_note)
                if row is not None:
                    rows.append(row)
            timings["pricing"] = time.perf_counter() - t_price
            timings["pricings"] = float(engine.n_pricings)
            unpriced = [r for r in rows if r["status"] == "unpriced"]
            no_pnl = [
                r for r in rows if r["status"] != "unpriced" and needs_pnl(r) and _nan(r["pnl"])
            ]
            if unpriced:
                reasons += sorted({str(r["reason"]) for r in unpriced})
            if no_pnl:
                reasons.append(prev_note or f"{len(no_pnl)} row(s) without P&L")
            status = "incomplete" if reasons else "ok"
            wall = time.perf_counter() - t_start
            timings["total"] = wall
            frame = pd.DataFrame(rows, columns=list(ROW_COLUMNS))
            fit = {
                "date": date,
                "config_hash": self.hash,
                "spot": st.spot,
                "close": st.close,
                "cache_key": st.key,
                "n_particles": st.spec.particle.n_particles,
                "horizon": st.spec.particle.horizon,
                "snapshot": _display(st.snapshot),
                "snapshot_sha256": st.snapshot_sha256,
                "source_sha256": st.source_sha256,
                "calibrated": date in calibrated,
                "timings": timings,
                **st.record,
            }
            done: dict[str, Any] = {
                "date": date,
                "status": status,
                "config_hash": self.hash,
                "source_sha256": st.source_sha256,
                "manifest_sha256": self.inputs.manifest_sha(date),
                "cache_key": st.key,
                "calibrated": calibrated,
                "n_rows": len(frame),
                "n_rows_without_pnl": len(no_pnl) + len(unpriced),
                "gaps": sorted(d for d in self.skipped if d < date),
                "previous_date": self.calendar[i - 1] if i > 0 else None,
                "previous_cache_key": None if prev is None else prev.key,
                "wall_s": wall,
                "timings": timings,
                "code_version": code_version(),
                "host": socket.gethostname(),
                "numba_threads": _numba_threads(),
                "created_utc": _utc_now(),
            }
            if reasons:
                done["incomplete_reasons"] = reasons
                done["command"] = self.command([date])
                done["leverage_missing"] = leverage_missing
            attempt, _, current = self._commit(
                date, frame, fit, done, used=st, prev_used=prev is not None
            )
        except DateFailure as exc:
            wall = time.perf_counter() - t_start
            msg = str(exc)
            if isinstance(exc, LeverageMissing):
                leverage_missing = True
            kept = self._write_failure(date, msg, wall, calibrated, timings)
            log.error("%s: FAILED (%s)%s", date, msg, kept)
            return DateOutcome(
                date, "failed", wall, calibrated, 0, msg, leverage_missing, current=not kept
            )
        except guard.CalibrationForbiddenError:
            raise  # a refused calibration is never a date's outcome: the run stops
        except Exception as exc:  # recorded with its type; a long run goes on to the next date
            wall = time.perf_counter() - t_start
            msg = f"{date}: unexpected {type(exc).__name__}: {exc}"
            log.exception("%s: FAILED", date)
            kept = self._write_failure(date, msg, wall, calibrated, timings)
            return DateOutcome(date, "failed", wall, calibrated, 0, msg, current=not kept)
        if not current:
            log.warning(
                "%s: %s attempt %s recorded; CURRENT keeps the attempt that protects more",
                date,
                status,
                attempt.id,
            )
        log.log(
            logging.WARNING if reasons else logging.INFO,
            "%s: %s, %d rows, %s, %.1f s (import %.1f, fit %.1f, calibration %.1f, pricing %.1f, "
            "%d pricings)%s",
            date,
            status.upper() if reasons else status,
            len(frame),
            f"calibrated {calibrated}" if calibrated else "leverage from the cache",
            wall,
            timings.get("import", 0.0),
            timings.get("fit", 0.0),
            timings.get("calibration", 0.0) + timings.get("calibration_previous", 0.0),
            timings["pricing"],
            int(timings["pricings"]),
            f": {'; '.join(reasons)}" if reasons else "",
        )
        return DateOutcome(
            date,
            status,
            wall,
            calibrated,
            len(frame),
            "; ".join(reasons),
            leverage_missing,
            current=current,
            attempt=attempt.id,
        )

    def _write_failure(
        self,
        date: str,
        error: str,
        wall: float,
        calibrated: Sequence[str],
        timings: Mapping[str, float],
    ) -> str:
        """Publish the failed attempt; it becomes current only when the current attempt protects
        nothing (:func:`pointer_may_move`).  Returns ``""`` or a note that the date kept its
        current attempt."""
        st = self._states.get(date)
        doc: dict[str, Any] = {
            "date": date,
            "status": "failed",
            "error": error,
            "config_hash": self.hash,
            "calibrated": list(calibrated),
            "n_rows": 0,
            "wall_s": wall,
            "timings": dict(timings),
            "command": self.command([date]),
            "code_version": code_version(),
            "created_utc": _utc_now(),
        }
        if st is not None:  # marked: later dates may have used its state
            doc["params"] = state_params(st.fit.params)
            doc["cache_key"] = st.key
        try:
            attempt, _, current = self._commit(date, None, None, doc)
        except Exception:  # the failure itself is reported by the caller
            log.exception("%s: the failed attempt could not be recorded", date)
            return ""
        if current:
            return ""
        return f"; recorded as {attempt.id}, CURRENT keeps the attempt that protects more"

    # -- one row -------------------------------------------------------------------------------

    def _base_row(self, trade: TradeInstance, date: str) -> dict[str, Any]:
        unit, scale = TRADE_UNITS[trade.spec.kind]
        row: dict[str, Any] = {c: np.nan for c in ROW_COLUMNS}
        row.update(
            date=date,
            trade_id=trade.trade_id,
            book=trade.book,
            kind=trade.spec.kind,
            inception=trade.inception,
            age=self.index[date] - self.index[trade.inception],
            unit=unit,
            scale=scale,
            maturity=trade.spec.maturity,
            config_hash=self.hash,
            reason="",
            pnl_method="none",
            pnl_note="",
            ladders_json="",
            realised_json="",
            extra_pricings=0.0,
            gaps=json.dumps(self.gaps(trade.inception, date)),
        )
        return row

    def trade_row(
        self,
        trade: TradeInstance,
        date: str,
        st: DateState,
        prev: DateState | None,
        engine: RiskEngine,
        prev_note: str,
    ) -> dict[str, Any] | None:
        t0 = time.perf_counter()
        i, j = self.index[date], self.index[trade.inception]
        try:
            built = self.built(trade)
            hist = self.history(trade.inception, date)
        except DateFailure as exc:
            row = self._base_row(trade, date)
            row.update(status="unpriced", reason=str(exc), pnl_note=f"unpriced: {exc}")
            row["seconds"] = time.perf_counter() - t0
            return row
        rep1 = replay(built.product, hist, dt.date.fromisoformat(date), discount=st.discount)
        today = dt.date.fromisoformat(date)
        flows_d = float(sum(f.amount for f in rep1.cash_flows if f.date == today))
        rep0: Replay | None = None
        if i > j:
            d0 = dt.date.fromisoformat(self.calendar[i - 1])
            rep0 = replay(built.product, hist, d0, discount=None if prev is None else prev.discount)
            if rep0.settled and float(getattr(rep0.result, "pay_time", 1.0)) <= 0.0:
                return None  # settled and paid before today: the trade is closed
        row = self._base_row(trade, date)
        row.update(
            strike=built.strike,
            vol_ko=built.vol_ko,
            flows=flows_d,
            flows_cum=float(sum(f.amount for f in rep1.cash_flows)),
        )
        state = dict(rep1.state)
        row["realised_json"] = _dumps(state)
        _realised_columns(row, built, state)
        v1 = self._value(rep1, st, engine)
        row["value"], row["value_stderr"] = v1
        if rep1.settled:
            row["status"] = (
                "paid" if float(getattr(rep1.result, "pay_time", 0.0)) <= 0 else "settled"
            )
            row["reason"] = str(getattr(rep1.result, "reason", ""))
        else:
            row["status"] = "inception" if i == j else "live"
        if i == j:
            row["pnl_note"] = "inception: no P&L"
        elif rep0 is None:
            row["pnl_note"] = "no previous date"
        elif prev is None:
            row["pnl_note"] = prev_note
        else:
            self._pnl(row, trade, built, hist, rep0, rep1, st, prev, engine, flows_d)
            if attributed(self.cfg, trade) and not _nan(row["pnl"]):
                self._cumulative(row, trade, built, rep1, st, engine)
        row["seconds"] = time.perf_counter() - t0
        return row

    def _cumulative(
        self,
        row: dict[str, Any],
        trade: TradeInstance,
        built: BuiltTrade,
        rep1: Replay,
        st: DateState,
        engine: RiskEngine,
    ) -> None:
        """``cum_pnl`` = V(date) − V(inception) + every cash flow paid so far (holder sign) and
        its DIRECT paired standard error: the date's value and the fresh trade at the inception
        state priced on common random numbers — one extra pricing (the date's leg is the value,
        already in the engine's memo); a settled trade's value is known cash, so the error is
        the inception price's.  The inception state and its leverage come from the cache (the
        date's own dependency record names the inception state)."""
        try:
            inc = self.state(trade.inception)
        except DateFailure as exc:
            row["pnl_note"] = f"{row['pnl_note']}; no cumulative P&L: {exc}".lstrip("; ")
            return
        p_inc = replay(
            built.product,
            self.history(trade.inception, trade.inception),
            dt.date.fromisoformat(trade.inception),
            discount=inc.discount,
        ).result
        if not isinstance(p_inc, Product):
            return
        flows = float(row["flows_cum"]) if not _nan(row["flows_cum"]) else 0.0
        n0 = engine.n_pricings
        try:
            if isinstance(rep1.result, Product):
                s = engine.paired(
                    "cum_pnl",
                    [
                        (rep1.result, st.state, "recalibrate", 1.0),
                        (p_inc, inc.state, "recalibrate", -1.0),
                    ],
                    unit="price",
                    size=0.0,
                    scheme="revaluation",
                )
                value, se = float(s.value), float(s.stderr)
            else:
                pr = engine.price(p_inc, inc.state)
                value, se = float(row["value"]) - float(pr.mean), float(pr.stderr)
        except CacheMissError as exc:
            row["pnl_note"] = (
                f"{row['pnl_note']}; no cumulative P&L: the inception leverage is not in the "
                f"cache ({exc})"
            ).lstrip("; ")
            return
        row["cum_pnl"] = value + flows
        row["cum_pnl_stderr"] = se
        row["extra_pricings"] = float(row["extra_pricings"] or 0.0) + (engine.n_pricings - n0)

    def _value(self, rep: Replay, st: DateState, engine: RiskEngine) -> tuple[float, float]:
        if isinstance(rep.result, Product):
            pr = engine.price(rep.result, st.state)
            return float(pr.mean), float(pr.stderr)
        return float(rep.result.value), 0.0

    def _pnl(
        self,
        row: dict[str, Any],
        trade: TradeInstance,
        built: BuiltTrade,
        hist: RealisedHistory,
        rep0: Replay,
        rep1: Replay,
        st: DateState,
        prev: DateState,
        engine: RiskEngine,
        flows_d: float,
    ) -> None:
        for b in BUCKETS:
            row[bucket_column(b)] = 0.0
            row[bucket_column(b) + "_stderr"] = 0.0
        row[bucket_column("cash_flows")] = flows_d
        p0, p1 = rep0.result, rep1.result
        v1 = float(row["value"])
        if not isinstance(p0, Product):  # settled before today, paid later: carry
            v0 = float(p0.value)
            row.update(price_0=v0, price_0_stderr=0.0)
            row["pnl"] = v1 - v0 + flows_d
            row["pnl_stderr"] = 0.0
            row[bucket_column("settlement")] = v1 - v0
            row["pnl_method"] = "settled"
            row["pnl_note"] = "settled trade: discount accretion of the known cash"
            return
        if not isinstance(p1, Product):  # settles today
            pr0 = engine.price(p0, prev.state)
            row.update(price_0=float(pr0.mean), price_0_stderr=float(pr0.stderr))
            row["pnl"] = v1 - float(pr0.mean) + flows_d
            row["pnl_stderr"] = float(pr0.stderr)
            row[bucket_column("settlement")] = v1 - float(pr0.mean)
            row[bucket_column("settlement") + "_stderr"] = float(pr0.stderr)
            row["pnl_method"] = "settlement"
            row["pnl_note"] = f"settled today: {getattr(p1, 'reason', '')}"
            return
        note = ""
        if attributed(self.cfg, trade):
            d1 = dt.date.fromisoformat(st.date)
            n0 = len(hist.dates) - 2
            held = RealisedHistory(hist.trade_date, hist.dates[: n0 + 1], hist.closes[: n0 + 1])
            p_theta = replay(
                built.product, held.extended(d1, hist.closes[n0]), d1, discount=prev.discount
            ).result
            if isinstance(p_theta, Product):
                try:
                    self._attribute(row, p0, p1, p_theta, st, prev, engine, flows_d)
                    return
                except (ArbitrageError, ValueError, NotImplementedError) as exc:
                    note = f"attribution failed ({type(exc).__name__}: {exc}); paired P&L"
                    log.warning("%s %s: %s", st.date, trade.trade_id, note)
            else:
                note = "the held-spot theta product settles; paired P&L"
        else:
            note = "valuation only (attribution.trades); paired P&L"
        s = engine.paired(
            "pnl",
            [(p1, st.state, "recalibrate", 1.0), (p0, prev.state, "recalibrate", -1.0)],
            unit="price",
            size=0.0,
            scheme="revaluation",
        )
        pr0 = engine.price(p0, prev.state)
        row.update(price_0=float(pr0.mean), price_0_stderr=float(pr0.stderr))
        row["pnl"] = float(s.value) + flows_d
        row["pnl_stderr"] = float(s.stderr)
        row[bucket_column("unattributed")] = float(s.value)
        row[bucket_column("unattributed") + "_stderr"] = float(s.stderr)
        row["pnl_method"] = "paired"
        row["pnl_note"] = note

    def _attribute(
        self,
        row: dict[str, Any],
        p0: Product,
        p1: Product,
        p_theta: Product,
        st: DateState,
        prev: DateState,
        engine: RiskEngine,
        flows_d: float,
    ) -> None:
        a = self.cfg.attribution
        pillars = tuple(float(x) for x in a["pillars"])
        bump = float(a["bump"])
        ex = explain(
            engine,
            p0,
            prev.state,
            st.state,
            dt=TRADING_DT,
            detail=str(a["detail"]),
            pillars=pillars,
            size=bump,
            mode=str(a["mode"]),
            product_1=p1,
            product_theta=p_theta,
        )
        n_before = engine.n_pricings
        greeks, errors, ladders = start_greeks(
            engine, ex, p0, p_theta, prev, st, pillars, bump, str(a["detail"])
        )
        row["extra_pricings"] = float(engine.n_pricings - n_before)
        row.update(
            price_0=float(ex.price_0), price_0_stderr=float(engine.price(p0, prev.state).stderr)
        )
        row["pnl"] = float(ex.total.value) + flows_d
        row["pnl_stderr"] = float(ex.total.stderr)
        row["pnl_method"] = "attributed"
        if abs(float(ex.price_1) - float(row["value"])) > 1e-12 * max(
            1.0, abs(float(row["value"]))
        ):
            row["pnl_note"] = f"attribution end price {ex.price_1!r} differs from the value"
        for name, value in ex.buckets().items():
            row[bucket_column(name)] = float(value)
        for name, se in errors.items():
            row[bucket_column(name) + "_stderr"] = float(se)
        # the residual's and the groups' stderrs: paired per path by explain (no extra pricing);
        # a root sum of squares only for an item without a path, flagged
        row[bucket_column("residual") + "_stderr"] = float(ex.residual_stderr)
        row["residual_paired"] = float(ex.residual_paired)
        for name in PAIRED_GROUPS:
            row[group_column(name)] = float(ex.group_stderrs.get(name, np.nan))
        if ex.unpaired:
            row["pnl_note"] = (
                f"{row['pnl_note']}; " if row.get("pnl_note") else ""
            ) + f"root-sum-of-squares stderrs (no per-path item): {', '.join(ex.unpaired)}"
        row[bucket_column("cash_flows")] = flows_d
        for s in ex.steps:
            row[f"s_{s.name}"] = float(s.actual)
            row[f"s_{s.name}_stderr"] = float(s.actual_stderr)
            row[f"s_{s.name}_explained"] = float(s.explained)
        for name, sens in greeks.items():
            row[f"g_{name}"] = float(sens.value)
            row[f"g_{name}_stderr"] = float(sens.stderr)
        row["ladders_json"] = _dumps(ladders) if ladders else ""


def _nan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def needs_pnl(row: Mapping[str, Any]) -> bool:
    """Whether a row should carry a P&L: a live, settled or paid trade after its inception."""
    return str(row.get("status")) in ("live", "settled", "paid")


def _realised_columns(row: dict[str, Any], built: BuiltTrade, state: Mapping[str, Any]) -> None:
    n = state.get("realised_returns")
    if n is not None:
        row["realised_returns"] = float(n)
        ss = state.get("realised_sum_sq")
        if ss is not None and int(n) > 0:
            # per-return annualisation: the product's (VKO, KO var: 252), else the trading grid's
            a = getattr(built.product, "annualisation", None)
            ann = float(a) if isinstance(a, (int, float)) else float(TRADING_DAYS_PER_YEAR)
            row["realised_vol"] = math.sqrt(ann * float(ss) / int(n))
    if "knocked_out" in state:
        row["knocked_out"] = float(bool(state["knocked_out"]))
    if "knocked_in" in state:
        row["knocked_in"] = float(bool(state["knocked_in"]))


def _numba_threads() -> int:
    try:
        import numba

        return int(numba.get_num_threads())  # type: ignore[no-untyped-call]
    except Exception:  # pragma: no cover - numba is a dependency
        return int(os.environ.get("NUMBA_NUM_THREADS", "0") or 0)


#: The placeholder a printed command uses for a store the user must name (shell-safe: it parses
#: as a path and is replaced by a new directory).
NEW_STORE = "NEW_STORE"


def backtest_argv(
    cfg: BacktestConfig,
    subcommand: str = "run",
    dates: Sequence[str] = (),
    *,
    extras: Sequence[str] = (),
    resume: bool = True,
    config: str | None = None,
    paths: Mapping[str, str] | None = None,
) -> list[str]:
    """**The one builder of every ``volsto-backtest`` command the module prints** — refusals,
    the stage-2 requirements, ``status`` hints, verdict reasons and the command stored with an
    incomplete or failed date: ``volsto-backtest <subcommand> <config>`` (``config``, else the
    config's source, cwd-relative), the ``--only-dates`` selection of a run, ALWAYS explicit
    ``--out``, ``--cache`` and ``--snapshots`` (``paths`` overrides them) — so the line runs as
    printed after a relocation and under ``VOLSTO_BACKTEST_REQUIRE_PATHS=1`` — ``--resume`` for
    a run line unless ``resume=False``, then ``extras``.  :func:`command_line` renders it."""
    src = cfg.source if config is None else config
    argv = ["volsto-backtest", subcommand, _cwd_path(_resolve(src)) if src else "<config>"]
    if dates and subcommand in ("run", "dry-run"):
        argv += ["--only-dates", *dates]
    where = {k: _cwd_path(cfg.path(k)) for k in ("out", "cache", "snapshots")}
    where.update(paths or {})
    for key in ("out", "cache", "snapshots"):
        argv += [f"--{key}", where[key]]
    if subcommand == "run" and resume:
        argv.append("--resume")
    return [*argv, *extras]


def command_line(argv: Sequence[str]) -> str:
    """A printed command: :func:`backtest_argv` shell-quoted (``shlex.split`` gives it back)."""
    return shlex.join(argv)


def backtest_command(
    cfg: BacktestConfig,
    subcommand: str = "run",
    dates: Sequence[str] = (),
    **kwargs: Any,
) -> str:
    """``command_line(backtest_argv(...))``."""
    return command_line(backtest_argv(cfg, subcommand, dates, **kwargs))


def elsewhere_command(cfg: BacktestConfig, dates: Sequence[str]) -> str:
    """The ``run`` line of ``dates`` into a new store :data:`NEW_STORE`, its snapshots inside it
    — never into the refused store's snapshots, which a config with other surface settings would
    re-import over."""
    return backtest_command(
        cfg, "run", dates, paths={"out": NEW_STORE, "snapshots": f"{NEW_STORE}/snapshots"}
    )


# --------------------------------------------------------------------------------------------
# start-of-period Greeks and bucket standard errors
# --------------------------------------------------------------------------------------------


def ladder_moves(
    surf0: ImpliedSurface, surf1: ImpliedSurface, pillars: Sequence[float]
) -> dict[str, np.ndarray]:
    """The per-pillar surface moves the attribution multiplies its ladders by: ATM vol, the
    90/110 skew and the 90/110 butterfly (``explain``'s definitions)."""
    ps = np.asarray(pillars, dtype=np.float64)
    k90, k110 = np.full(ps.size, K90), np.full(ps.size, K110)

    def quantities(s: ImpliedSurface) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        atm = np.asarray(s.atm_vol(ps), dtype=np.float64)
        v90 = np.asarray(s.implied_vol_k(k90, ps), dtype=np.float64)
        v110 = np.asarray(s.implied_vol_k(k110, ps), dtype=np.float64)
        return atm, v90 - v110, 0.5 * (v90 + v110) - atm

    a0, s0, f0 = quantities(surf0)
    a1, s1, f1 = quantities(surf1)
    return {"atm": a1 - a0, "skew": s1 - s0, "fly": f1 - f0}


def start_greeks(
    engine: RiskEngine,
    ex: Explain,
    p0: Product,
    p_theta: Product,
    prev: DateState,
    st: DateState,
    pillars: tuple[float, ...],
    bump: float,
    detail: str,
) -> tuple[dict[str, Sensitivity], dict[str, float], dict[str, Any]]:
    """The Greeks :func:`explain` priced at the start of the period (read back through the
    frozen-leverage wrapper from the engine's memo), each bucket's standard error — the paired
    one of the per-path sum of its items (:attr:`Explain.bucket_stderrs`: a ladder bucket's
    pillars are correlated), |move| × the Greek's for a single item — and the ladders'
    per-pillar values."""
    w = FrozenLeverageEngine(engine, prev.state)
    s0 = prev.state
    steps = {s.name for s in ex.steps}
    g: dict[str, Sensitivity] = {}
    err: dict[str, float] = {}
    ladders: dict[str, Any] = {}
    if "spot" in steps:
        d, gm = delta_gamma(w, p0, s0, "sticky_moneyness", bump)
        g["delta"], g["gamma"] = d, gm
        ds = st.spot - prev.spot
        err["spot.delta"] = abs(ds) * d.stderr
        err["spot.gamma"] = 0.5 * ds * ds * gm.stderr
    if "rates" in steps:
        rs = curve_move_sensitivities(w, p0, s0, st.state)
        g["rho"], g["repo"] = rs["rho"], rs["repo"]  # the P&L of the whole move, first order
        err["rates.rho"] = rs["rho"].stderr
        err["rates.repo"] = rs["repo"].stderr
    if "surface" in steps:
        moves = ladder_moves(surface_of(s0), surface_of(st.state), pillars)
        if detail == "parallel":
            v = parallel_vega(w, p0, s0, "recalibrated", bump)
            g["vega"] = v
            err["surface.parallel_vega"] = abs(float(moves["atm"].mean())) / 0.01 * v.stderr
        else:
            vt = vega_T(w, p0, s0, pillars, bump, with_tents=True)
            sk = skew_T(w, p0, s0, pillars, bump)
            cv = curvature_T(w, p0, s0, pillars, bump)
            for name, sens, move in (
                ("surface.vega_T", vt.tents, moves["atm"]),
                ("surface.skew_T", sk.entries, moves["skew"]),
                ("surface.curvature_T", cv.entries, moves["fly"]),
            ):
                err[name] = float(
                    math.sqrt(sum((abs(m) / 0.01 * e.stderr) ** 2 for e, m in zip(sens, move)))
                )
                ladders[name] = {
                    "pillars": list(pillars),
                    "value": [e.value for e in sens],
                    "stderr": [e.stderr for e in sens],
                    "move": [float(m) for m in move],
                }
            g["vega"] = vt.parallel
    if "params" in steps:
        m0, m1 = prev.spec.model, st.spec.model
        for name in PARAMS:
            p_0, p_1 = float(getattr(m0, name)), float(getattr(m1, name))
            if param_moved(p_0, p_1):
                ps_ = parameter_sensitivity(w, p0, s0, name)
                g[f"d_{name}"] = ps_
                err[f"params.{name}"] = abs(p_1 - p_0) * ps_.stderr
    if "time" in steps:
        th = theta(w, p0, s0, TRADING_DT, aged=p_theta)
        g["theta"], g["theta_decay"] = th.total, th.decay
        g["theta_carry"], g["theta_roll_down"] = th.carry, th.roll_down
        err["time.decay"] = TRADING_DT * th.decay.stderr
        err["time.carry"] = TRADING_DT * th.carry.stderr
    for s in ex.steps:
        if s.name == "recalibration":
            err["recalibration"] = float(s.actual_stderr)
    # the buckets' errors from explain's per-path sums (a ladder's pillars share the seed: the
    # root sum of squares above is only the fallback of an item without a path)
    err.update({k: float(v) for k, v in ex.bucket_stderrs.items() if k in err})
    return g, err, ladders


# --------------------------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------------------------


@dataclass
class Probe:
    """What the projection's probe measured (module docstring)."""

    config_hash: str
    dates: tuple[str, ...]
    threads: int
    import_s: float
    import_source: str
    fit_s: float
    build_s: float
    probe_paths: tuple[int, int]
    #: kind-and-maturity label → (seconds at the two probe sizes)
    pricing: dict[str, tuple[float, float]]
    #: fixed-trade spec id → (pricings, states) of one attribution
    attribution: dict[str, tuple[int, int]]
    #: distinct model states of one attribution date (all attributed trades together)
    states_per_date: int
    wall_s: float
    #: distinct ξ₀ strips among those states, the previous date's excluded (already stripped
    #: by that date's valuation in the same process: a memo hit, ``volsto.calibration.cache.
    #: XI0_MEMO``)
    strips_per_date: int = 0
    #: a model build whose strip is a memo hit (surface construction and the kernel)
    build_warm_s: float = 0.0
    created_utc: str = ""

    def builds_s(self, attributed: bool) -> float:
        """Model-build seconds of a date: every state at the warm cost, every new strip at
        the difference (one state, today's, on a date without attribution)."""
        if not attributed:
            return self.build_s
        strip_s = max(self.build_s - self.build_warm_s, 0.0)
        return self.states_per_date * self.build_warm_s + self.strips_per_date * strip_s

    def pricing_s(self, label: str, n_paths: int) -> float:
        """Seconds of one pricing at ``n_paths``: the affine fit through the two probe sizes
        (slope floored at 0, intercept at 0)."""
        hi, lo = self.probe_paths
        t_hi, t_lo = self.pricing[label]
        slope = max((t_hi - t_lo) / (hi - lo), 0.0) if hi != lo else t_hi / hi
        intercept = max(t_hi - slope * hi, 0.0)
        return float(intercept + slope * n_paths)

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Probe:
        d = dict(data)
        d["dates"] = tuple(d["dates"])
        d["probe_paths"] = tuple(d["probe_paths"])
        d["pricing"] = {k: (float(v[0]), float(v[1])) for k, v in d["pricing"].items()}
        d["attribution"] = {k: (int(v[0]), int(v[1])) for k, v in d["attribution"].items()}
        return cls(**d)


class _StateRecorder(DryRunEngine):
    """The attribution dry run that also keeps the states it was asked to price (the
    projection counts their ξ₀ strips)."""

    def __init__(self, sim: SimConfig, base: RiskState, max_halvings: int) -> None:
        super().__init__(sim, base, max_halvings)
        self.states: list[tuple[RiskState, str]] = []

    def priced(self, product: Product, state: RiskState, mode: str = "recalibrate") -> Any:
        self.states.append((state, mode))
        return super().priced(product, state, mode)


def _strip_key(state: RiskState, mode: str, base_spot: float, horizon: float) -> str | None:
    """The ξ₀ memo key of the model build a request implies (a ``"model"`` build strips the
    state at the base spot)."""
    spec = state.spec
    if mode == "model":
        spec = dataclasses.replace(spec, market=dataclasses.replace(spec.market, spot=base_spot))
    t_max = min(spec.surface.max_maturity, max(horizon + 1.0, 5.0))
    return XI0_MEMO.key(spec, t_max)


def trade_label(spec: TradeSpec) -> str:
    return f"{spec.kind}@{spec.maturity:g}"


def flat_leverage(spec: CalibrationSpec, forward_curve: ForwardCurve) -> LeverageFunction:
    """L = 1 on a grid of the production size (2001 log-moneyness points): the pricing cost
    of an LSV does not depend on the leverage values."""
    k = spec.particle.leverage_dk * (np.arange(2001) - 1000)
    times = np.array([0.0, spec.particle.horizon])
    return LeverageFunction(times, k, np.ones((times.size, k.size)), forward_curve)


def _bound_here(run: BacktestRun, date: str) -> bool:
    """Whether the date's configured snapshot exists and is bound to this config's inputs."""
    path = run.snapshot_path(date)
    try:
        sha = run.inputs.file_sha(date)
        return path.is_file() and snapshot_bound(run.cfg, path, sha, run.inputs.manifest_sha(date))
    except (DateFailure, OSError):
        return False


def run_probe(
    run: BacktestRun, dates: Sequence[str], stored_import_s: Sequence[float] = ()
) -> Probe:
    """Measure the per-date costs on the first two consecutive ``dates`` that can be marked
    (module docstring; at most :data:`PROBE_ATTEMPTS` dates are tried, a failing date restarts
    the pair; with a single markable date the attribution is not counted).  Nothing is
    calibrated; snapshots missing from the store are imported into a temporary directory.
    Raises :class:`DateFailure` when no tried date can be marked."""
    t_start = time.perf_counter()
    cfg = run.cfg
    with tempfile.TemporaryDirectory(prefix="volsto-probe-") as tmp:
        probe_run = BacktestRun(cfg, allow_calibrate=False, snapshots=run.snapshots)
        imported: list[float] = []
        tried = list(dates[:PROBE_ATTEMPTS])
        if not all(_bound_here(probe_run, d) for d in tried):
            # never (re)import into the configured snapshots from a probe: they may be another
            # config's (a dry run of another config) or in use by a running writer
            probe_run.snapshots = Path(tmp)
        states: list[DateState] = []
        errors: list[str] = []
        for d in tried:
            try:
                st = probe_run.state(d)
            except DateFailure as exc:
                errors.append(str(exc))
                log.warning("probe: %s", exc)
                states = []  # the pair must be consecutive
                continue
            states.append(st)
            if st.timings["import"] > 0:
                imported.append(st.timings["import"])
            if len(states) == 2:
                break
        if not states:
            raise DateFailure(f"no probe date can be marked: {'; '.join(errors)}")
        if imported:
            import_s, import_source = (
                float(np.mean(imported)),
                f"measured on {len(imported)} date(s)",
            )
        elif stored_import_s:
            import_s = float(np.median(stored_import_s))
            import_source = f"median of {len(stored_import_s)} stored import timing(s)"
        else:
            import_s = IMPORT_S_FALLBACK
            import_source = f"constant {IMPORT_S_FALLBACK:g} s (measured 2026-09-16, 2022-07-27)"
        fit_s = float(np.mean([s.timings["fit"] for s in states]))
        a = states[0]
        memo_state = XI0_MEMO.enabled
        XI0_MEMO.enabled = False
        try:
            t0 = time.perf_counter()
            _, _, kernel = build_market(a.spec)
            build_s = time.perf_counter() - t0
        finally:
            XI0_MEMO.enabled = memo_state
        warm: list[float] = []
        for _ in range(2):  # the first call fills the memo (when it is on), the second hits
            t0 = time.perf_counter()
            build_market(a.spec)
            warm.append(time.perf_counter() - t0)
        build_warm_s = warm[-1] if XI0_MEMO.enabled else build_s
        model = LSV(kernel, flat_leverage(a.spec, a.forward_curve))
        hi = min(cfg.n_paths, PROBE_PATHS_MAX)
        hi -= hi % 2
        lo = max(2, hi // PROBE_PATHS_RATIO)
        lo -= lo % 2
        specs = {trade_label(t): t for t in (*cfg.fixed, *cfg.rolling)}
        pricing: dict[str, tuple[float, float]] = {}
        for label, spec in specs.items():
            built = build_product(spec, a.close, a.surface, a.discount)
            times = []
            for n in (hi, lo):
                sim = dataclasses.replace(run.sim, n_paths=n, chunk_size=min(run.sim.chunk_size, n))
                mc = MonteCarlo(sim)
                prod = built.product.with_discount(a.discount)
                best = math.inf
                for _ in range(PROBE_REPEATS):  # the best of the repeats: load noise
                    t1 = time.perf_counter()
                    mc.price(prod, model, grid=mc.build_grid([prod], model))
                    best = min(best, time.perf_counter() - t1)
                times.append(best)
            pricing[label] = (times[0], times[1])
            log.info(
                "probe: %s %.3f s at %d paths, %.3f s at %d", label, times[0], hi, times[1], lo
            )
        attribution: dict[str, tuple[int, int]] = {}
        all_states: set[tuple[str, str, Any]] = set()
        strips: set[str | None] = set()
        if len(states) == 2 and str(cfg.attribution["trades"]) != "none":
            b = states[1]
            a_cfg = cfg.attribution
            d1 = dt.date.fromisoformat(b.date)
            hist = RealisedHistory(
                dt.date.fromisoformat(a.date),
                (dt.date.fromisoformat(a.date), d1),
                (a.spot, b.spot),
            )
            held = RealisedHistory(hist.trade_date, hist.dates[:1], hist.closes[:1])
            for spec in cfg.fixed:
                built = build_product(spec, a.close, a.surface, a.discount)
                p1 = replay(built.product, hist, d1, discount=b.discount).result
                pt = replay(
                    built.product, held.extended(d1, a.close), d1, discount=a.discount
                ).result
                if not isinstance(p1, Product) or not isinstance(pt, Product):
                    continue
                dry = _StateRecorder(run.sim, a.state, int(a_cfg["max_halvings"]))
                explain(
                    dry,
                    built.product,
                    a.state,
                    b.state,
                    dt=TRADING_DT,
                    detail=str(a_cfg["detail"]),
                    pillars=tuple(float(x) for x in a_cfg["pillars"]),
                    size=float(a_cfg["bump"]),
                    mode=str(a_cfg["mode"]),
                    product_1=p1,
                    product_theta=pt,
                )
                trade_states = {(k, m, x) for _, k, m, x in dry.requests}
                all_states |= trade_states
                strips |= {_strip_key(st, m, a.spot, cfg.horizon) for st, m in dry.states}
                attribution[spec.id] = (len(dry.requests), len(trade_states))
    return Probe(
        config_hash=cfg.content_hash(),
        dates=tuple(s_.date for s_ in states),
        threads=_numba_threads(),
        import_s=import_s,
        import_source=import_source,
        fit_s=fit_s,
        build_s=build_s,
        probe_paths=(hi, lo),
        pricing=pricing,
        attribution=attribution,
        states_per_date=len(all_states),
        wall_s=time.perf_counter() - t_start,
        strips_per_date=len(strips - {_strip_key(a.state, "recalibrate", a.spot, cfg.horizon)}),
        build_warm_s=build_warm_s,
        created_utc=_utc_now(),
    )


@dataclass
class DateCost:
    date: str
    import_s: float = 0.0
    fit_s: float = 0.0
    calibration_s: float = 0.0
    builds_s: float = 0.0
    attribution_s: float = 0.0
    valuation_s: float = 0.0
    boundary_s: float = 0.0
    cumulative_s: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.import_s
            + self.fit_s
            + self.calibration_s
            + self.builds_s
            + self.attribution_s
            + self.valuation_s
            + self.boundary_s
            + self.cumulative_s
        )


@dataclass
class Projection:
    """The projected wall clock (module docstring)."""

    config_hash: str
    n_calendar: int
    todo: list[str]
    costs: list[DateCost]
    probe: Probe
    calibration_s: float
    calibration_source: str
    misses_assumed: int
    snapshots_missing: int
    shards: dict[int, tuple[float, float]]
    requested_shard: tuple[int, int] | None

    @property
    def total(self) -> float:
        return float(sum(c.total for c in self.costs))

    def component(self, name: str) -> float:
        return float(sum(getattr(c, name) for c in self.costs))


def _hours(s: float) -> str:
    if s >= 3600:
        return f"{s / 3600:.2f} h"
    if s >= 60:
        return f"{s / 60:.1f} min"
    return f"{s:.1f} s"


#: The calibration charged when the cache manifest has no timing: the ``volsto-precompute``
#: fallback, 140 s at 8·10⁵ particles over 3y (the m6-tag median is 138 s), scaled linearly in
#: particles × calibration steps (``scaled_calibration_s``).
CALIBRATION_REFERENCE_S: Final[float] = 140.0
CALIBRATION_REFERENCE_PARTICLES: Final[int] = 800_000
CALIBRATION_REFERENCE_HORIZON: Final[float] = 3.0
#: Floor of a scaled calibration: the particle pass of a 2·10⁴-particle, 1y calibration took
#: 3.0–3.6 s at 2 threads (2026-09-16) — below ~10⁵ particles the regression window floor
#: (2000 particles) makes the cost more than linear in the particle count.
CALIBRATION_MIN_S: Final[float] = 3.0


def calibration_steps(spec: CalibrationSpec, horizon: float) -> int:
    """Steps of the calibration grid over ``horizon`` (the particle pass's own grid)."""
    return int(TimeGrid.build([float(horizon)], spec.sim.dt_max).dts.size)


def scaled_calibration_s(
    cache: LeverageCache, spec: CalibrationSpec, n_particles: int, horizon: float
) -> tuple[float, str]:
    """Projected seconds of one calibration and where they come from: the cache manifest's
    median at the configured particles and horizon (the current code tag preferred); else the
    median of the nearest (particles × steps) group rescaled linearly in particles × steps;
    else :data:`CALIBRATION_REFERENCE_S` rescaled the same way; never below
    :data:`CALIBRATION_MIN_S`."""
    work = n_particles * calibration_steps(spec, horizon)
    m = cache.manifest()
    needed = {"wall_time", "n_particles", "horizon"}
    if not m.empty and needed <= set(m.columns):
        m = m[np.isfinite(m["wall_time"].to_numpy(dtype=float))]
        if "code_tag" in m.columns and (m["code_tag"] == CALIBRATION_CODE_TAG).any():
            m = m[m["code_tag"] == CALIBRATION_CODE_TAG]
    if m.empty or not needed <= set(m.columns):
        ref_work = CALIBRATION_REFERENCE_PARTICLES * calibration_steps(
            spec, CALIBRATION_REFERENCE_HORIZON
        )
        sec = CALIBRATION_REFERENCE_S * work / ref_work
        return max(sec, CALIBRATION_MIN_S), (
            f"no timing in the cache manifest: the {CALIBRATION_REFERENCE_S:g} s reference at "
            f"{CALIBRATION_REFERENCE_PARTICLES} particles, {CALIBRATION_REFERENCE_HORIZON:g}y, "
            f"scaled by particles x steps ({sec:.2f} s), floor {CALIBRATION_MIN_S:g} s"
        )
    groups = m.groupby(["n_particles", "horizon"])["wall_time"].median()
    best = None
    for idx, med in groups.items():
        n_g, h_g = cast(tuple[Any, Any], idx)
        w = int(n_g) * calibration_steps(spec, float(h_g))
        dist = abs(math.log(w / work))
        if best is None or dist < best[0]:
            best = (dist, int(n_g), float(h_g), float(med), w)
    assert best is not None
    _, n, h, med, w = best
    count = int(((m["n_particles"] == n) & (np.isclose(m["horizon"], h))).sum())
    if n == n_particles and abs(h - horizon) < 1e-12:
        return med, f"cache manifest median of {count} calibrations at {n} particles, {h:g}y"
    sec = med * work / w
    return max(sec, CALIBRATION_MIN_S), (
        f"cache manifest median {med:.1f} s of {count} calibrations at {n} particles, {h:g}y, "
        f"scaled by particles x steps ({sec:.2f} s), floor {CALIBRATION_MIN_S:g} s"
    )


class _CostModel:
    """Per-date costs of one config from a probe (:func:`project`)."""

    def __init__(self, run: BacktestRun, probe: Probe, *, calibrate: bool = True) -> None:
        self.run = run
        self.probe = probe
        cfg = run.cfg
        if calibrate:
            self.cal_s, self.cal_source = scaled_calibration_s(
                run.cache, run.base, cfg.n_particles, cfg.horizon
            )
        else:
            self.cal_s, self.cal_source = 0.0, "--no-calibrate: a missing leverage fails its date"
        self.keys: dict[str, str] = {}
        for d in run.calendar:
            f = run.store.current_fit(d)
            if f is not None and f.get("config_hash") == run.hash and f.get("cache_key"):
                self.keys[d] = str(f["cache_key"])
        self.cached = {k for k in set(self.keys.values()) if run.cache.has_key(k)}

    def calibration(self, date: str) -> float:
        """A calibration unless the date's key is known to be cached."""
        return 0.0 if self.keys.get(date) in self.cached else self.cal_s

    def date(self, date: str) -> DateCost:
        """The cost of a date computed after its predecessor in the same process."""
        run, probe, cfg = self.run, self.probe, self.run.cfg
        i = run.index[date]
        c = DateCost(date, fit_s=probe.fit_s, calibration_s=self.calibration(date))
        attributes = i > 0 and bool(probe.attribution)
        c.builds_s = probe.builds_s(attributes)
        fallback = max((v[0] for v in probe.attribution.values()), default=2)
        for t in run.live(date):
            per = probe.pricing_s(trade_label(t.spec), cfg.n_paths)
            if i == run.index[t.inception]:
                c.valuation_s += per
            elif attributed(cfg, t):
                n = probe.attribution.get(t.spec.id, (fallback, 0))[0]
                c.attribution_s += n * per
                if run.index[t.inception] < i - 1:  # the day after inception: a memo hit
                    c.cumulative_s += per  # the fresh trade at the inception state
            else:
                c.valuation_s += 2 * per  # today's value and the previous one (paired P&L)
        return c

    def block_start(self, date: str) -> float:
        """What a process starting at ``date`` adds: the previous date's fit and leverage, and
        the fit and leverage of every earlier inception of an attributed trade live at ``date``
        (the cumulative P&L prices the fresh trade at its inception state)."""
        run = self.run
        i = run.index[date]
        if i == 0:
            return 0.0
        prev = run.calendar[i - 1]
        s = self.probe.fit_s + self.calibration(prev)
        inceptions = {
            t.inception
            for t in run.live(date)
            if attributed(run.cfg, t) and run.index[t.inception] < i - 1
        }
        for d in sorted(inceptions):
            s += self.probe.fit_s + self.calibration(d)
        return s

    def imports(self, last: str) -> int:
        """Snapshots a process computing up to ``last`` must import first."""
        cal = self.run.calendar
        return sum(
            1 for d in cal[: self.run.index[last] + 1] if not self.run.snapshot_path(d).is_file()
        )


def project(
    run: BacktestRun,
    todo: Sequence[str],
    *,
    probe: Probe,
    requested_shard: tuple[int, int] | None = None,
    selection: Sequence[str] | None = None,
    blocks_over: Sequence[str] | None = None,
    calibrate: bool = True,
) -> Projection:
    """The projected wall clock of computing ``todo`` (module docstring): one process, and
    ``n`` shards of contiguous blocks run in parallel (the slowest block, and the sum) over
    ``selection`` (default ``todo``; under ``--shard`` the unsharded selection, so the table
    describes the whole run while the totals describe this shard); the blocks are cut from
    ``blocks_over`` (the ``--only-dates`` selection before ``--resume``; default the
    calendar), as ``--shard`` cuts them."""
    model = _CostModel(run, probe, calibrate=calibrate)
    whole = set(selection if selection is not None else todo)
    cut = list(blocks_over) if blocks_over is not None else list(run.calendar)
    todo_set = set(todo)
    ordered = [d for d in run.calendar if d in todo_set]
    costs: list[DateCost] = []
    previous: str | None = None
    for d in ordered:
        c = model.date(d)
        i = run.index[d]
        if i > 0 and previous != run.calendar[i - 1]:
            c.boundary_s = model.block_start(d)
        costs.append(c)
        previous = d
    missing = model.imports(ordered[-1]) if ordered else 0
    if costs:
        costs[0].import_s = missing * probe.import_s
    by_date = {c.date: c for c in costs}
    for d in whole - set(by_date):
        by_date[d] = model.date(d)
    shards: dict[int, tuple[float, float]] = {}
    for n in sorted({*SHARD_COUNTS, *([requested_shard[1]] if requested_shard else [])}):
        walls = []
        for k in range(1, n + 1):
            block = [d for d in shard_block(cut, k, n) if d in whole]
            if not block:
                continue
            w = sum(by_date[d].total - by_date[d].boundary_s - by_date[d].import_s for d in block)
            w += model.block_start(block[0]) + model.imports(block[-1]) * probe.import_s
            walls.append(w)
        shards[n] = (max(walls, default=0.0), float(sum(walls)))
    misses = sum(1 for c in costs if c.calibration_s > 0) + sum(
        1 for c in costs if c.boundary_s > probe.fit_s
    )
    return Projection(
        run.hash,
        len(run.calendar),
        list(ordered),
        costs,
        probe,
        model.cal_s,
        model.cal_source,
        misses,
        missing,
        shards,
        requested_shard,
    )


def format_projection(p: Projection, cfg: BacktestConfig) -> str:
    """The projection block printed before any work."""
    n = len(p.costs)
    pr = p.probe
    a = cfg.attribution
    lines = [
        f"PROJECTED WALL CLOCK — backtest {cfg.name} (config hash {p.config_hash[:12]}): "
        f"{n} of {p.n_calendar} calendar dates to compute",
        f"  probe: {' -> '.join(pr.dates)}, pricing at {pr.probe_paths[0]} and "
        f"{pr.probe_paths[1]} paths scaled affinely to {cfg.n_paths} paths, "
        f"NUMBA threads {pr.threads}, probe wall clock {pr.wall_s:.1f} s",
        f"  attribution: mode {a['mode']}, detail {a['detail']}, trades {a['trades']}, "
        f"{pr.states_per_date} model states per attributed date, {pr.strips_per_date} new "
        f"xi0 strips (a build {pr.build_s:.2f} s with its strip, {pr.build_warm_s:.2f} s on a "
        "memo hit)",
    ]
    for spec in cfg.fixed:
        if spec.id in pr.attribution:
            np_, ns = pr.attribution[spec.id]
            per = pr.pricing_s(trade_label(spec), cfg.n_paths)
            lines.append(
                f"    {spec.id}: {np_} pricings x {per:.2f} s = {np_ * per:.0f} s per date "
                f"({ns} states)"
            )
    rows = [
        (
            "import (missing snapshots)",
            "import_s",
            f"{p.snapshots_missing} x {pr.import_s:.2f} s, {pr.import_source}",
        ),
        ("marking fit", "fit_s", f"{pr.fit_s:.2f} s per date (+1 per shard block start)"),
        (
            "leverage calibration",
            "calibration_s",
            f"{p.calibration_s:.1f} s each, {p.calibration_source}; "
            f"{p.misses_assumed} miss(es) assumed",
        ),
        (
            "shard-boundary calibration",
            "boundary_s",
            "previous date and earlier inceptions of attributed trades, per block start",
        ),
        ("model builds", "builds_s", "states x warm build + new strips x strip seconds"),
        ("attribution pricings", "attribution_s", "fixed book (or as configured)"),
        ("valuation pricings", "valuation_s", "inception + paired P&L of the other trades"),
        (
            "cumulative P&L pricings",
            "cumulative_s",
            "1 pricing per attributed trade per date (the fresh trade at its inception state, "
            "paired with the date's value; none the day after inception)",
        ),
    ]
    lines.append(f"  {'component':<28}{'per date':>12}{'total':>12}  source")
    for label, attr, src in rows:
        tot = p.component(attr)
        lines.append(f"  {label:<28}{_hours(tot / max(n, 1)):>12}{_hours(tot):>12}  {src}")
    lines.append(
        f"  {'TOTAL (one process)':<28}{_hours(p.total / max(n, 1)):>12}{_hours(p.total):>12}"
    )
    scope = "this shard's dates" if p.requested_shard else "the selected dates"
    lines.append(f"  the totals above cover {scope}; the shard lines cover the whole selection:")
    for k, (slowest, total) in sorted(p.shards.items()):
        lines.append(
            f"  {k} shard(s): wall clock {_hours(slowest)} (slowest contiguous block), "
            f"{_hours(total)} of process time"
        )
    lines.append(
        "  assumptions: every process runs at the probe's per-process speed (thread scaling is "
        "weak: 1.70x at 12 threads, M10 Part 0); knock-outs and autocalls are not predicted; a "
        "date whose leverage key is unknown is charged a calibration"
    )
    return "\n".join(lines)


def load_or_probe(
    run: BacktestRun, todo: Sequence[str], *, refresh: bool = False, persist: bool = True
) -> Probe:
    """``<out>/probe.json`` when it matches the config hash and the thread count, else a new
    probe — written to the store only with ``persist`` (``run`` passes ``False`` for a dry run
    that ``run`` would refuse), when the store exists and its header, if any, names this
    config."""
    path = run.store.root / PROBE_NAME
    data = None if refresh else _read_json(path)
    if data is not None:
        try:
            probe = Probe.from_mapping(data)
        except (KeyError, TypeError, ValueError):
            probe = None
        if (
            probe is not None
            and probe.config_hash == run.hash
            and probe.threads == _numba_threads()
        ):
            return probe
    start = run.index[todo[0]] if todo else 0
    stored = []
    for d in run.calendar:
        f = run.store.current_fit(d)
        if f and f.get("timings", {}).get("import"):
            stored.append(float(f["timings"]["import"]))
    probe = run_probe(run, run.calendar[max(start - 1, 0) :], stored)
    header = run.store.header()
    if (
        persist
        and run.store.root.is_dir()
        and (header is None or header.get("config_hash") == run.hash)
    ):
        _write_json(path, probe.to_mapping())  # never into another config's store
    return probe


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


#: The range separator of ``--only-dates`` (``2022-07-01..2022-08-05``, both ends included).
RANGE_SEP = ".."


def _split_dates(values: Sequence[str] | None) -> list[str] | None:
    """``--only-dates`` items (space or comma separated): ``YYYY-MM-DD`` or ``A..B``."""
    if values is None:
        return None
    out: list[str] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if not part:
                continue
            if RANGE_SEP in part:
                a, _, b = part.partition(RANGE_SEP)
                lo, hi = _iso(a.strip(), "--only-dates"), _iso(b.strip(), "--only-dates")
                if hi < lo:
                    raise ConfigError(f"--only-dates: the range {part} ends before it starts")
                out.append(f"{lo}{RANGE_SEP}{hi}")
            else:
                out.append(_iso(part, "--only-dates"))
    return out


def expand_dates(calendar_dates: Sequence[str], only: Sequence[str]) -> list[str]:
    """The calendar dates named by ``--only-dates`` items (single dates must be calendar
    dates; a range ``A..B`` takes the calendar dates inside ``[A, B]`` and must hold one)."""
    chosen: set[str] = set()
    unknown: list[str] = []
    for item in only:
        if RANGE_SEP in item:
            lo, _, hi = item.partition(RANGE_SEP)
            inside = [d for d in calendar_dates if lo <= d <= hi]
            if not inside:
                raise ConfigError(f"--only-dates: no calendar date in {item}")
            chosen.update(inside)
        elif item in calendar_dates:
            chosen.add(item)
        else:
            unknown.append(item)
    if unknown:
        raise ConfigError(f"--only-dates: {sorted(unknown)} are not calendar dates of the config")
    return [d for d in calendar_dates if d in chosen]


def selection_of(run: BacktestRun, only: Sequence[str] | None) -> list[str]:
    """The calendar dates ``--only-dates`` names (the whole calendar without it)."""
    return list(run.calendar) if only is None else expand_dates(run.calendar, only)


def resume_skips(v: Verdict) -> bool:
    """Whether ``--resume`` leaves a date alone: confirmed ``done`` or ``skipped``, or an ``ok``
    outcome that only waits for an unstored dependency (a shard boundary: it is confirmed or
    found stale once that date is stored)."""
    if v.settled:
        return True
    return v.status == "pending" and v.doc is not None and v.doc.get("status") == "ok"


def select_dates(
    run: BacktestRun,
    *,
    shard: tuple[int, int] | None,
    only: Sequence[str] | None,
    resume: bool,
    limit: int | None,
) -> tuple[list[str], list[str]]:
    """``(todo, skipped)``: the ``--only-dates`` selection (the calendar without it), the
    shard's contiguous block of that selection, the resume filter (:func:`resume_skips` on the
    :class:`Ledger` verdicts), then the limit.  The blocks are cut before ``--resume`` and
    ``--limit``, so a shard names the same dates on every run of the same selection.  A pending
    date that waits for a date of ``todo`` is kept: :func:`cmd_run` checks it again when its
    turn comes."""
    dates = selection_of(run, only)
    if shard is not None:
        dates = shard_block(dates, *shard)
    skipped: list[str] = []
    if resume:
        keep: list[str] = []
        for d in dates:
            v = run.ledger.verdict(d)
            if resume_skips(v) and not set(v.waits) & set(keep):
                skipped.append(d)
            else:
                keep.append(d)
        dates = keep
    if limit is not None:
        dates = dates[: max(limit, 0)]
    return dates, skipped


def _with_previous(run: BacktestRun, dates: Sequence[str]) -> list[str]:
    out = set(dates)
    for d in dates:
        p = run.ledger.previous(d)
        if p is not None:
            out.add(p)
    return sorted(out)


def hash_refusals(run: BacktestRun, todo: Sequence[str]) -> list[str]:
    """The dates of ``todo`` and their previous dates whose stored outcome (verified, or still
    in the flat layout) was computed under another config hash
    (:func:`computed_under_another_config`): recomputing next to them would mix configs
    (``--force`` accepts that)."""
    return [
        d
        for d in _with_previous(run, todo)
        if computed_under_another_config(run.ledger.verdict(d), run.hash)
    ]


def _results_key(v: Verdict) -> str | None:
    if v.stored is None or v.stored_status not in RESULT_STATUSES:
        return None
    lev = (v.stored.get("record") or {}).get("leverage")
    key = lev.get("key") if isinstance(lev, Mapping) else v.stored.get("cache_key")
    return str(key) if key else None


def leverage_refusals(run: BacktestRun, todo: Sequence[str]) -> dict[str, str]:
    """``{date: why}`` for ``--no-calibrate``, read through the :class:`Ledger` verdicts: a date
    of ``todo`` whose current results name a leverage the configured cache lacks (recomputing it
    could only fail); the previous date of a date of ``todo`` — with or without stored
    results of its own — when its leverage is lacking and either it has stored results or the
    stored P&L of the date used it (recomputing the date would lose its P&L).  The previous
    date's key comes from its own results, else from the date's ``previous_cache_key``, else
    (a store of an older volsto-backtest) from marking it, which never calibrates."""
    out: dict[str, str] = {}
    for d in todo:
        v = run.ledger.verdict(d)
        key = _results_key(v)
        if key is not None and not run.cache.has_key(key):
            out[d] = f"{d} ({key[:12]}, its stored results)"
        p = run.ledger.previous(d)
        if p is None or p in out:
            continue
        pkey = _results_key(run.ledger.verdict(p))
        why = f"the previous date of {d}, with stored results"
        if pkey is None and key is not None:
            assert v.stored is not None
            used = bool(((v.stored.get("record") or {}).get("previous") or {}).get("leverage"))
            if not used:
                continue
            why = f"used by the stored P&L of {d}; {p} has no stored results"
            pkey = v.stored.get("previous_cache_key")
            if pkey is None and not _bound_here(run, p):
                # marking would import first: a refusal must not write
                out[p] = f"{p} (its leverage key is unknown until its snapshot is imported; {why})"
                continue
            if pkey is None:
                try:
                    pkey = run.state(p).key
                except DateFailure as exc:
                    out[p] = f"{p} (cannot be marked: {exc}; {why})"
                    continue
        if pkey is not None and not run.cache.has_key(str(pkey)):
            out[p] = f"{p} ({str(pkey)[:12]}, {why})"
    return out


def _refuse(message: str) -> None:
    """A refusal: logged and printed to stderr (the CLI user sees it whatever the logging)."""
    log.error("REFUSED: %s", message)
    print(f"REFUSED: {message}", file=sys.stderr, flush=True)


def _refusal_command(run: BacktestRun, dates: Sequence[str], *extras: str) -> str:
    return backtest_command(run.cfg, "run", dates, extras=extras)


def up_front_refusal(
    run: BacktestRun, todo: Sequence[str], *, force: bool, no_calibrate: bool
) -> str:
    """Both refusals of ``run`` (``""`` when none), read through the :class:`Ledger` verdicts —
    of the flat layout too (:meth:`Ledger._flat_verdict`), so they come before its migration —
    each with runnable commands.  Pure: nothing is written."""
    if no_calibrate:
        # a wrong --cache must not cost stored results or their P&L
        lacking = leverage_refusals(run, todo)
        if lacking:
            dates = sorted(lacking)
            # the printed line must pass the next refusal too
            extra = ("--force",) if force or hash_refusals(run, dates) else ()
            return (
                f"the cache {_display(run.cache.root)} lacks the leverage of {len(lacking)} "
                f"date(s): {'; '.join(lacking[d] for d in dates)}. Under --no-calibrate "
                "recomputing the dates to compute could only fail or lose their P&L. Recalibrate "
                f"them with: {_refusal_command(run, dates, *extra)} — or rerun with --cache "
                "pointing at the cache that holds them"
            )
    stale = hash_refusals(run, todo)
    if stale and not force:
        return (
            f"{len(stale)} date(s) to compute or before them hold results computed under "
            f"another config hash ({', '.join(stale[:4])}{' ...' if len(stale) > 4 else ''}): "
            "recomputing next to them would mix configs. Recompute them under this config with: "
            f"{_refusal_command(run, todo, '--force')} — or keep them and write elsewhere "
            f"(replace {NEW_STORE}): "
            f"{elsewhere_command(run.cfg, todo)}"
        )
    return ""


def cmd_run(args: argparse.Namespace, *, dry: bool) -> int:
    t0 = time.perf_counter()
    cfg = load_backtest_config(args.config).with_paths(
        out=args.out, cache=args.cache, snapshots=args.snapshots
    )
    shard = parse_shard(args.shard) if args.shard else None
    no_calibrate = bool(getattr(args, "no_calibrate", False))
    run = BacktestRun(cfg, allow_calibrate=not (dry or no_calibrate))
    only = _split_dates(args.only_dates)
    force = bool(getattr(args, "force", False))

    def select() -> tuple[list[str], list[str]]:
        return select_dates(run, shard=shard, only=only, resume=args.resume, limit=args.limit)

    # every refusal before ANY write (header, migration, skips, adoption, probe, imports)
    why = foreign_pointer_refusal(run.store) or header_refusal(cfg, run.store, force=force)
    if why and not dry:
        _refuse(why)
        return EXIT_REFUSED
    # skip_date: the calendar is final before the shard blocks are cut (a pure check)
    chosen = selection_of(run, only)
    newly_skipped = run.new_skips(chosen[-1]) if chosen else []
    todo, resumed = select()
    if not dry:
        why = up_front_refusal(run, todo, force=force, no_calibrate=no_calibrate) or (
            migration_refusal(run.store, run.ledger, rebinding=force)
        )
        if why:
            _refuse(why)
            return EXIT_REFUSED
        check_writable(run.store, run.ledger)
        if force:  # --force: this config takes the store over before migrating it
            bind_header(cfg, run.store)
        if run.migrate():
            print(f"migrated {len(run.migration)} date(s) from the flat layout into attempts")
            chosen = selection_of(run, only)
            newly_skipped = run.new_skips(chosen[-1]) if chosen else []
        bind_header(cfg, run.store)  # after the migration: a refused one leaves no header
        if newly_skipped:
            run.write_skips(newly_skipped)
        # adoption after the skips are written (the calendar they define is the one judged)
        adopted = adopt(run.store, run.ledger, selection_of(run, only))
        if adopted:
            print(f"adopted a published attempt for {len(adopted)} date(s): {', '.join(adopted)}")
        todo, resumed = select()
    else:
        # the dry run judges what run would refuse (the per-date config check included); it
        # then persists nothing into the store
        why = (
            why
            or up_front_refusal(run, todo, force=force, no_calibrate=False)
            or migration_refusal(run.store, run.ledger, rebinding=force)
        )
        if why:
            print(f"dry run: a run would be refused: {why}")
    selection = todo
    if shard is not None:
        selection, _ = select_dates(run, shard=None, only=only, resume=args.resume, limit=None)
    blocks_over = selection_of(run, only)
    if run.skipped:
        print(
            f"data.missing_close: skip_date dropped {len(run.skipped)} date(s) from the calendar: "
            + "; ".join(f"{d} ({r})" for d, r in sorted(run.skipped.items()))
            + (" (dry run: not recorded)" if dry and newly_skipped else "")
        )
    if resumed:
        pending = [d for d in resumed if run.ledger.verdict(d).status == "pending"]
        waits = sorted({w for d in pending for w in run.ledger.verdict(d).waits})
        print(
            f"resume: {len(resumed) - len(pending)} date(s) done and still matching"
            + (
                f", {len(pending)} pending (their results verify; they wait for "
                f"{', '.join(waits)}, not in this selection)"
                if pending
                else ""
            )
            + ", skipped"
        )
    if not todo:
        print(
            f"nothing to compute ({len(resumed)} done); wall clock "
            f"{time.perf_counter() - t0:.1f} s; recalibrated: no"
        )
        return EXIT_OK
    proj: Projection | None = None
    try:
        probe = load_or_probe(run, todo, refresh=dry, persist=not why)
        proj = project(
            run,
            todo,
            probe=probe,
            requested_shard=shard,
            selection=selection,
            blocks_over=blocks_over,
            calibrate=not no_calibrate,
        )
        print(format_projection(proj, cfg), flush=True)
    except DateFailure as exc:
        print(f"PROJECTED WALL CLOCK unavailable: {exc}", flush=True)
    if dry:
        print(
            f"dry run: nothing computed; wall clock {time.perf_counter() - t0:.1f} s; "
            "recalibrated: no"
        )
        return EXIT_OK
    ctx = guard.calibration_forbidden() if no_calibrate else contextlib.nullcontext()
    outcomes: list[DateOutcome] = []
    rechecked: list[str] = []
    with ctx:
        run.resolve_closes(todo[-1])
        for d in todo:
            if args.resume and resume_skips(run.ledger.verdict(d)):
                rechecked.append(d)  # confirmed by the dates recomputed before it
                continue
            outcomes.append(run.run_date(d))
    calibrated = sorted({c for o in outcomes for c in o.calibrated})
    by_status: dict[str, list[DateOutcome]] = {}
    for o in outcomes:
        by_status.setdefault(o.status, []).append(o)
    bad = [o for o in outcomes if o.status != "ok"]
    wall = time.perf_counter() - t0
    print(
        f"backtest {cfg.name}: {len(by_status.get('ok', []))} date(s) ok, "
        f"{len(by_status.get('incomplete', []))} incomplete, "
        f"{len(by_status.get('failed', []))} failed, "
        f"{len(resumed) + len(rechecked)} skipped (resume"
        + (f"; {len(rechecked)} confirmed during the run" if rechecked else "")
        + f"); {len(calibrated)} leverage calibration(s); wall clock {wall:.1f} s (projected "
        f"{'n/a' if proj is None else f'{proj.total:.1f} s'}); "
        f"recalibrated: {'yes' if calibrated else 'no'}"
    )
    for o in bad:
        print(f"  {o.status.upper()} {o.date}: {o.error}")
    for o in outcomes:
        if not o.current:
            print(
                f"  KEPT {o.date}: this {o.status} attempt is recorded next to CURRENT, which "
                "keeps the attempt that protects more (see status)"
            )
    if bad:
        print(f"  recompute them with: {run.command([o.date for o in bad])}")
        return EXIT_REFUSED if any(o.leverage_missing for o in bad) else EXIT_FAILED
    return EXIT_OK


def _stage1_run(args: argparse.Namespace) -> BacktestRun:
    """The run of a stage-1 maintenance command (nothing written yet)."""
    cfg = load_backtest_config(args.config).with_paths(
        out=args.out, cache=args.cache, snapshots=args.snapshots
    )
    return BacktestRun(cfg, allow_calibrate=False)


def _migrated(run: BacktestRun) -> int | None:
    """Migrate a flat store (after its refusals); ``EXIT_REFUSED`` when refused."""
    try:
        run.migrate()
    except RefusedError as exc:
        _refuse(str(exc))
        return EXIT_REFUSED
    if run.migration:
        print(f"migrated {len(run.migration)} date(s) from the flat layout into attempts")
    return None


def _staging_summary(store: BacktestStore) -> dict[str, list[Path]]:
    """The staging directories by owner: ``dead`` (this host, writer gone: ``gc`` removes
    them), ``live`` (this host, writer running) and ``other`` (another host: never removed
    here)."""
    out: dict[str, list[Path]] = {"dead": [], "live": [], "other": []}
    for p in store.leftovers()["staging"]:
        tag = p.name[len(STAGING_PREFIX) :].split("-")[0]
        if tag != _host_tag():
            out["other"].append(p)
        else:
            out["live" if _staging_owner_alive(p) else "dead"].append(p)
    return out


def cmd_status(args: argparse.Namespace) -> int:
    run = _stage1_run(args)
    cfg = run.cfg
    refused = _migrated(run)
    if refused is not None:
        return refused
    counts: dict[str, int] = {}
    wall = 0.0
    cal = 0
    lines = []
    for d, v in run.ledger.verdicts().items():
        counts[v.status] = counts.get(v.status, 0) + 1
        doc = v.stored or {}
        wall += float(doc.get("wall_s") or 0.0)
        cal += len(doc.get("calibrated") or [])
        if args.verbose or v.status != "done":
            rows = doc.get("n_rows", "-")
            att = "" if v.attempt is None else f"  [{v.attempt.id[:24]}]"
            note = "; ".join(x for x in (v.note, v.legacy_unverified) if x)
            note = f" ({note})" if note else ""
            lines.append(
                f"  {d}  {v.status:<10} rows {rows!s:>3}  "
                f"wall {float(doc.get('wall_s') or 0.0):7.1f} s  "
                f"calibrated {doc.get('calibrated', [])}  {v.reason}{note}{att}"
            )
    legacy = [d for d, v in run.ledger.verdicts().items() if v.legacy_unverified]
    header = run.store.header()
    print(
        f"backtest {cfg.name}: store {_display(run.store.root)} "
        f"(header hash {str((header or {}).get('config_hash'))[:12]}, config hash "
        f"{run.hash[:12]}); {len(run.vendor_calendar)} vendor dates: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        + f"; stored wall clock {_hours(wall)}; {cal} calibration(s) recorded"
        + ("" if args.verbose else " (done dates listed with -v)")
    )
    for line in lines:
        print(line)
    if legacy:
        more = " ..." if len(legacy) > 6 else ""
        print(
            f"  {len(legacy)} date(s) with results carry a {LEGACY_UNVERIFIED}: "
            f"{', '.join(legacy[:6])}{more} (their leverage is checked by key and completeness "
            f"only; {LEGACY_RECOMPUTE}: {backtest_command(cfg, 'run', legacy, resume=False)})"
        )
    gc_line = backtest_command(cfg, "gc")
    _, problem = _migration_journal(run.store)
    if problem:
        print(f"  {problem}")
    left = run.store.leftovers()
    staging = _staging_summary(run.store)
    n_attempts = sum(len(run.store.attempts(d)) for d in run.store.dates())
    n_current = sum(1 for d in run.store.dates() if run.store.pointer(d) is not None)
    if n_attempts > n_current or left["pointer"]:
        print(
            f"  {n_attempts - n_current} non-current attempt(s) and {len(left['pointer'])} "
            f"pointer temporary(ies) ({gc_line} removes those of done dates and the "
            "temporaries, waiting for running writers)"
        )
    if staging["dead"] or staging["live"]:
        print(
            f"  {len(staging['dead'])} staging directory(ies) of finished writers on this host "
            f"({gc_line} removes them) and {len(staging['live'])} of running ones"
        )
    if staging["other"]:
        hosts = sorted({p.name[len(STAGING_PREFIX) :].split("-")[0] for p in staging["other"]})
        print(
            f"  {len(staging['other'])} staging directory(ies) of writers on other hosts "
            f"({', '.join(hosts)}): gc here keeps them (it cannot tell whether they still run); "
            f"run {gc_line} on that host, or remove them by hand once its writers "
            "have finished: "
            + " ".join(_display(p) for p in staging["other"][:3])
            + (" ..." if len(staging["other"]) > 3 else "")
        )
    finished = counts.get("done", 0) + counts.get("skipped", 0)
    return EXIT_OK if finished == len(run.vendor_calendar) else EXIT_FAILED


def cmd_migrate(args: argparse.Namespace) -> int:
    run = _stage1_run(args)
    refused = _migrated(run)
    if refused is not None:
        return refused
    moved = sum(len(e.get("attempts", [])) for e in run.migration.values())
    print(
        f"backtest {run.cfg.name}: migrated {len(run.migration)} date(s) ({moved} attempt(s)) "
        f"from the flat layout into attempts under {_display(run.store.dates_root)}"
        + (f"; recorded in {MIGRATIONS_NAME}" if run.migration else "")
    )
    return EXIT_OK


def cmd_gc(args: argparse.Namespace) -> int:
    run = _stage1_run(args)
    why = gc_refusal(run.store, run.ledger)
    if why:
        _refuse(why)
        return EXIT_REFUSED
    check_writable(run.store, run.ledger, run.store.dates(), attempts=True)
    refused = _migrated(run)
    if refused is not None:
        return refused
    try:
        counts = collect_garbage(run.store, run.ledger)
    except RefusedError as exc:
        _refuse(str(exc))
        return EXIT_REFUSED
    except OSError as exc:  # a partly read-only store: nothing current was touched
        raise ConfigError(
            f"gc on {run.store.root}: {exc}: make the store writable (chmod -R u+w) and run gc "
            "again"
        ) from exc
    print(
        f"backtest {run.cfg.name}: removed {counts['attempts']} non-current attempt(s) of done "
        f"dates, {counts['staging']} staging directory(ies) of finished writers on this host, "
        f"{counts['pointer']} pointer temporary(ies) and {counts['flat']} migrated flat "
        f"leftover(s); kept {counts['kept']} attempt(s) of dates that are not done and "
        f"{counts['staging_kept']} staging directory(ies) of running writers or of other hosts "
        f"(run gc on those hosts); skipped {counts['symlinks']} symbolic link(s)"
    )
    return EXIT_OK


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_FAILED, f"{self.prog}: error: {message}\n")


#: Opt-in guard (``=1``): every command refuses unless ``--out``, ``--cache`` and
#: ``--snapshots`` are given — for harnesses that must never fall back on a config's paths.
REQUIRE_PATHS_ENV = "VOLSTO_BACKTEST_REQUIRE_PATHS"

EXIT_CODES_HELP = (
    "exit codes: 0 success; 1 failure (a date failed, or bad arguments); 2 refused (nothing "
    "written: another config's store, a CURRENT of a newer volsto-backtest, a leverage missing "
    "under --no-calibrate).  With VOLSTO_BACKTEST_REQUIRE_PATHS=1 in the environment, a command "
    "without all of --out, --cache and --snapshots (non-blank) is refused (exit 2) before "
    "anything is read or written: set it for agents and scripted runs, so the config's paths.* "
    "are never used."
)

_SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "run": (
        "compute dates (the only M10 path that calibrates)",
        "Compute the selected dates into the store: every refusal is checked before anything is "
        "written; then the header is bound, a flat store migrated, skips recorded, published "
        "attempts adopted, and each date marked, priced, attributed and published as an "
        "immutable attempt.",
    ),
    "dry-run": (
        "print the projected wall clock only",
        "Print what `run` would compute and its projected wall clock (one process and shards). "
        "Writes nothing but snapshots and probe.json (the latter only into a store of this "
        "config).",
    ),
    "status": (
        "per-date verdicts of the store",
        "Print the verdict of every vendor date (done dates only with -v), the stored wall "
        "clock and the leftovers of writers (staging directories of this and other hosts). "
        "Migrates a flat store first (refused for another config). Exit 0 when every date is "
        "done or skipped, 1 otherwise, 2 when refused.",
    ),
    "migrate": (
        "move a store of the flat layout into attempts",
        "Move the flat layout (dates/<date>/{rows.parquet, fit.json, done.json}) into immutable "
        "attempts and CURRENT pointers, journalled in migrations.json (an interrupted "
        "migration finishes on the next open). Refused (exit 2) for a store of another config.",
    ),
    "gc": (
        "remove non-current attempts and leftovers of finished writers",
        "Remove the non-current attempts of done dates, pointer temporaries and staging "
        "directories of finished writers on this host (other hosts' are kept and counted). "
        "Refused (exit 2) unless the store header names this config.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="volsto-backtest",
        description="Rolling date-by-date backtest (M10 Part 3)",
        epilog=EXIT_CODES_HELP,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str) -> argparse.ArgumentParser:
        short, long = _SUBCOMMANDS[name]
        q = sub.add_parser(name, help=short, description=long, epilog=EXIT_CODES_HELP)
        q.add_argument("config", help="the backtest config (configs/backtest/*.yaml)")
        q.add_argument("--out", help="the per-date store (overrides paths.out)")
        q.add_argument("--cache", help="the leverage cache (overrides paths.cache)")
        q.add_argument("--snapshots", help="the snapshot directory (overrides paths.snapshots)")
        return q

    for name in ("run", "dry-run"):
        q = add(name)
        q.add_argument(
            "--shard",
            metavar="I/N",
            help="the I-th of N contiguous blocks of the selected dates (--only-dates, else the "
            "calendar); blocks are cut before --resume and --limit",
        )
        q.add_argument(
            "--resume",
            action="store_true",
            help="leave dates alone that are done (or pending an unstored dependency)",
        )
        q.add_argument("--limit", type=int, metavar="N", help="compute at most N dates")
        q.add_argument(
            "--only-dates",
            nargs="+",
            metavar="DATE",
            help="dates YYYY-MM-DD or ranges YYYY-MM-DD..YYYY-MM-DD (comma or space separated)",
        )
        q.add_argument("-v", "--verbose", action="store_true", help="debug logging")
        if name == "run":
            q.add_argument(
                "--no-calibrate",
                action="store_true",
                help="read leverages from the cache only (calibration forbidden; refused when "
                "the cache lacks a leverage stored results need)",
            )
            q.add_argument(
                "--force",
                action="store_true",
                help="accept a store or dates computed under another config hash: rebind the "
                "header and recompute next to them",
            )
    for name in ("status", "migrate", "gc"):
        q = add(name)
        q.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="list done dates too (status) and debug logging",
        )
    return p


def missing_path_flags(args: argparse.Namespace) -> list[str]:
    """The path flags a parsed command lacks or gives blank (what
    ``VOLSTO_BACKTEST_REQUIRE_PATHS=1`` refuses)."""
    given = {k: str(getattr(args, k, None) or "").strip() for k in ("out", "cache", "snapshots")}
    return [f"--{k}" for k, v in given.items() if not v]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("volsto").setLevel(level)
    if os.environ.get(REQUIRE_PATHS_ENV) == "1":
        lacking = missing_path_flags(args)
        if lacking:
            _refuse(
                f"{REQUIRE_PATHS_ENV}=1: pass {', '.join(lacking)} (the config's paths.* are "
                "not used, so a command whose overrides were dropped writes nothing)"
            )
            return EXIT_REFUSED
    try:
        if args.command == "status":
            return cmd_status(args)
        if args.command == "migrate":
            return cmd_migrate(args)
        if args.command == "gc":
            return cmd_gc(args)
        return cmd_run(args, dry=args.command == "dry-run")
    except RefusedError as exc:
        _refuse(str(exc))
        return EXIT_REFUSED
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        if getattr(args, "verbose", False):
            raise
        return EXIT_FAILED


# --------------------------------------------------------------------------------------------
# stage 2: the study module
# --------------------------------------------------------------------------------------------

TITLE = "Rolling backtest: marks, P&L attribution and realised dynamics"
QUESTION = (
    "Day by day over the history, how does the marked book move, which risks explain its P&L, "
    "and how do the marked SSR and the fitted parameters compare with what was realised?"
)
REQUIRED_PARAMS: tuple[str, ...] = ("backtest", "store")
OPTIONAL_PARAMS: tuple[str, ...] = ()


def validate_params(params: Mapping[str, Any]) -> None:
    """``backtest``: a backtest config path (repository-relative or absolute); ``store``: the
    per-date store as an artefact directory **relative to the study's ``outputs`` root** (the
    runner's artefact convention), or ``null`` for the config's ``paths.out`` (which must then
    lie under the outputs root)."""
    if not isinstance(params["backtest"], str) or not params["backtest"]:
        raise ConfigError("params.backtest must be the path of a backtest config")
    store = params["store"]
    if store is not None:
        if not isinstance(store, str) or not store:
            raise ConfigError("params.store must be a directory under the outputs root, or null")
        p = Path(store)
        if p.is_absolute() or ".." in p.parts:
            raise ConfigError(
                f"params.store {store!r} must be relative to the study's outputs root "
                "(pass --outputs to move the root)"
            )


def _study_config(ctx: StudyContext) -> tuple[BacktestConfig, Path, str]:
    """The backtest config (its ``paths.out`` set to the store, its ``paths.snapshots`` moved
    with the store when the config keeps them inside it, its cache the study's), the store
    directory and its id relative to the outputs root."""
    cfg = load_backtest_config(_resolve(str(ctx.params["backtest"])))
    store = ctx.params["store"]
    if store is not None:
        try:
            inside = cfg.path("snapshots").relative_to(cfg.path("out"))
        except ValueError:
            inside = None
        cfg = cfg.with_paths(out=ctx.outputs_root / str(store))
        if inside is not None:
            cfg = cfg.with_paths(snapshots=ctx.outputs_root / str(store) / inside)
    cfg = cfg.with_paths(cache=ctx.cache_root)
    root = cfg.path("out").resolve()
    try:
        rel = root.relative_to(ctx.outputs_root.resolve())
    except ValueError as exc:
        raise ConfigError(
            f"the backtest store {root} is not under the study's outputs root "
            f"{ctx.outputs_root} (artefact ids are relative to it): set outputs or params.store"
        ) from exc
    return cfg, root, rel.as_posix()


def _study_ledger(ctx: StudyContext) -> tuple[BacktestConfig, str, Ledger]:
    """The backtest config, the store's artefact id and the :class:`Ledger` of the store with
    the study's cache (the configured snapshots)."""
    cfg, root, rel = _study_config(ctx)
    dates = calendar(cfg)
    if not dates:
        raise ConfigError(f"the backtest config {cfg.source} has no calendar dates")
    ledger = Ledger(
        cfg,
        BacktestStore(root),
        InputIndex(cfg, dates),
        snapshots=cfg.path("snapshots"),
        cache=LeverageCache(cfg.path("cache")),
    )
    return cfg, rel, ledger


def unconfirmed(
    ctx: StudyContext, cfg: BacktestConfig, rel: str, verdicts: Mapping[str, Verdict]
) -> MissingRequirements | None:
    """Every date the ledger does not confirm, as requirements with the commands that produce
    them — the same judgements as ``run``'s refusals: a flat store of this config is migrated
    first; every date to recompute goes into ONE ``run`` line, which carries ``--force`` exactly
    when ``run`` would refuse it otherwise (:func:`computed_under_another_config` on the dates
    and their previous dates, or a store header of another config); the dates a pending one
    waits for are computed; an ``unsettled`` date (writers were active) gets no run line, and a
    ``CURRENT`` of a newer volsto-backtest — which every writer of this version refuses — makes
    that refusal the only command."""
    bad = {d: v for d, v in verdicts.items() if v.status in UNCONFIRMED_STATUSES}
    if not bad:
        return None
    h = cfg.content_hash()
    store = BacktestStore(cfg.path("out"))
    header = store.header()
    flat = [d for d, v in bad.items() if v.flat]
    # a store run --force must rebind: another config's header, or no header and a flat date
    # of another config (which migrate refuses, as run does)
    foreign_header = (header is not None and header.get("config_hash") != h) or (
        header is None and any(computed_under_another_config(bad[d], h) for d in flat)
    )
    migrate_first = bool(flat) and not foreign_header
    rerun = sorted(
        {
            d
            for d, v in bad.items()
            if v.status in ("stale", "missing")
            and not v.foreign
            and (not v.flat or foreign_header or computed_under_another_config(v, h))
        }
        | {w for v in bad.values() if v.status == "pending" for w in v.waits}
    )
    around = set(rerun) | {p for d in rerun if (p := _previous(verdicts, d)) is not None}
    force = foreign_header or any(
        computed_under_another_config(verdicts[e], h) and not verdicts[e].flat
        for e in around
        if e in verdicts
    )
    reloc = _relocated(cfg, ctx)
    cmd_migrate = backtest_command(reloc, "migrate") if migrate_first else ""
    cmd_run = (
        backtest_command(reloc, "run", rerun, extras=["--force"] if force else []) if rerun else ""
    )
    cmd_status = backtest_command(reloc, "status")
    refused = foreign_pointer_refusal(store)
    if refused:
        cmd_migrate = cmd_run = ""
    reqs: list[Requirement] = []
    for d, v in bad.items():
        what = {
            "stale": f"backtest {cfg.name} date {d} is stale: {v.reason}",
            "missing": f"backtest {cfg.name} date {d} is not computed ({v.reason})",
            "pending": f"backtest {cfg.name} date {d} is unconfirmed: {v.reason}",
            "unsettled": f"backtest {cfg.name} date {d} could not be read consistently while "
            f"writers changed the store ({v.reason}): rerun the study once they finish",
        }[v.status]
        if v.foreign:
            cmd = f"# {d}: {v.reason} — use the volsto-backtest that wrote it"
        elif refused:
            cmd = f"# {refused}"
        elif v.status == "unsettled":
            cmd = cmd_status
        elif d in flat and migrate_first and d not in rerun:
            cmd = cmd_migrate
        else:
            cmd = cmd_run or cmd_status
        reqs.append(Requirement("artefact", f"{rel}/{DATES_DIR}/{d}/{CURRENT_NAME}", what, cmd))
    commands = [c for c in (cmd_migrate, cmd_run) if c]
    if cmd_migrate:
        commands[1:] = [f"# after the migration: {c}" for c in commands[1:]]
    unsettled = sorted(d for d, v in bad.items() if v.status == "unsettled")
    if unsettled:
        commands.append(
            f"# {len(unsettled)} date(s) were being written ({', '.join(unsettled[:3])}): "
            f"rerun the study once the writers finish ({cmd_status})"
        )
    commands += sorted({str(r.command) for r in reqs if str(r.command).startswith("# ")})
    return MissingRequirements(reqs, commands)


#: The verdict statuses stage 2 cannot read.
UNCONFIRMED_STATUSES: tuple[str, ...] = ("stale", "missing", "pending", "unsettled")


def _previous(verdicts: Mapping[str, Verdict], date: str) -> str | None:
    """The previous date of the effective calendar the verdicts describe (skips left out)."""
    cal = [d for d, v in verdicts.items() if v.status != "skipped"]
    if date not in cal:
        return None
    i = cal.index(date)
    return cal[i - 1] if i > 0 else None


def requirements(ctx: StudyContext) -> list[Requirement]:
    """One artefact per calendar date: its ``CURRENT`` pointer (whose checksum pins the attempt
    and every file hash); any date the :class:`Ledger` does not confirm raises
    :class:`~volsto.studies.runner.MissingRequirements` with its command (exit 2)."""
    cfg, rel, ledger = _study_ledger(ctx)
    verdicts = ledger.verdicts()
    missing = unconfirmed(ctx, cfg, rel, verdicts)
    if missing is not None:
        raise missing
    reloc = _relocated(cfg, ctx)
    return [
        Requirement(
            "artefact",
            f"{rel}/{DATES_DIR}/{d}/{CURRENT_NAME}",
            f"backtest {cfg.name} date {d} ({v.status}, attempt {v.files.name})",
            backtest_command(reloc, "run", [d]),
        )
        for d, v in verdicts.items()
    ]


def _relocated(cfg: BacktestConfig, ctx: StudyContext) -> BacktestConfig:
    """``cfg`` with its source set to the study's config path (for the command's overrides)."""
    return dataclasses.replace(cfg, source=_display(_resolve(str(ctx.params["backtest"]))))


def compute(ctx: StudyContext) -> Results:
    cfg, rel, ledger = _study_ledger(ctx)
    verdicts = ledger.verdicts()  # again: the store may have changed since requirements()
    missing = unconfirmed(ctx, cfg, rel, verdicts)
    if missing is not None:
        raise missing
    dates = list(ledger.vendor_calendar)
    h = cfg.content_hash()
    ctx.record("backtest_inputs_verified", True)
    legacy = [d for d in dates if verdicts[d].legacy_unverified]
    ctx.record(
        "backtest_legacy_unverified",
        {"what": LEGACY_UNVERIFIED, "count": len(legacy), "dates": legacy},
    )
    dones = {d: dict(verdicts[d].doc or {}) for d in dates}
    ok = [d for d in dates if verdicts[d].has_results]
    # read what was verified: the bytes the verdicts checked, parsed here (never re-read)
    frames = [verdicts[d].verified.frame() for d in ok]
    rows = (
        pd.concat([f for f in frames if not f.empty], ignore_index=True)
        if any(not f.empty for f in frames)
        else pd.DataFrame(columns=list(ROW_COLUMNS))
    )
    # a store written before a column existed (cum_pnl, the paired group errors) reads NaN
    rows = rows.reindex(columns=list(dict.fromkeys([*ROW_COLUMNS, *rows.columns])))
    fits = {d: verdicts[d].verified.json(FIT_NAME) for d in ok}
    for d in dates:  # pin the CURRENT bytes compute() read (not the ones requirements() saw)
        sha = verdicts[d].pointer_sha256
        if not sha:
            raise ConfigError(f"{d}: the verdict read no {CURRENT_NAME} ({verdicts[d].status})")
        ctx.artefacts[f"{rel}/{DATES_DIR}/{d}/{CURRENT_NAME}"] = sha
    live_counts = {
        d: len(live_trades(cfg, ledger.calendar(), ledger.trades(), d))
        for d in dates
        if dones[d].get("status") == "failed" and d in ledger.calendar()
    }
    ctx.record("n_paths", cfg.n_paths)
    ctx.record("backtest_config", cfg.source)
    ctx.record("backtest_config_hash", h)
    ctx.record("backtest_particles", cfg.n_particles)
    seed = int(cfg.section("pricing")["seed"])
    if ctx.seed("pricing") != seed:
        raise ConfigError(
            f"seeds.pricing {ctx.seed('pricing')} of the study differs from the backtest's "
            f"pricing seed {seed} ({cfg.source}): the study documents the seed stage 1 used"
        )
    for d in ok:
        key = fits[d].get("cache_key")
        if isinstance(key, str) and key:
            ctx.cache_keys.setdefault(
                key,
                {
                    "key": key,
                    "what": f"backtest {cfg.name} leverage of {d} (stage 1; not read by the study)",
                    "n_particles": int(fits[d].get("n_particles") or cfg.n_particles),
                },
            )
    b = ResultsBuilder()
    _setup_results(b, cfg, dates, dones, rows, ctx.mode)
    b.add_exact(
        "setup",
        "backtest",
        "n_legacy_unverified",
        float(len(legacy)),
        unit="",
        source=SRC,
        note=", ".join(legacy),
    )

    _window_results(b, cfg, ok, fits)
    _missing_pnl_results(b, dates, dones, rows, live_counts)
    _trade_results(b, rows)
    _pnl_results(b, rows)
    _fit_results(b, cfg, ok, fits)
    _ssr_results(b, cfg, ok, fits)
    _vko_results(b, rows)
    _cost_results(b, dates, dones, fits)
    return b.build()


SRC = "computed"


def _setup_results(
    b: ResultsBuilder,
    cfg: BacktestConfig,
    dates: Sequence[str],
    dones: Mapping[str, Mapping[str, Any]],
    rows: pd.DataFrame,
    mode: str,
) -> None:
    status_of = {d: str(dones[d].get("status")) for d in dates}
    marked = [d for d in dates if status_of[d] in ("ok", "incomplete")]
    skipped = [d for d in dates if status_of[d] == "skipped"]
    variance = [t for t in (*cfg.fixed, *cfg.rolling) if t.kind in VARIANCE_KINDS]
    fit_cfg = cfg.stability_fit_config()
    fit_text = ", ".join(
        f"{k} {cfg.section('stability')['fit'][k]}" for k in STABILITY_FIT_REQUIRED
    )
    data = cfg.section("data")
    a = cfg.attribution
    m = cfg.section("marking")
    items: list[tuple[str, float, str]] = [
        ("n_dates", float(len(dates)), f"{dates[0]} to {dates[-1]}"),
        (
            "n_ok",
            float(len(marked)),
            ", ".join(f"{d} {status_of[d]}" for d in dates if status_of[d] != "ok"),
        ),
        ("n_skipped", float(len(skipped)), ", ".join(skipped)),
        (
            "data",
            float(len(dates)),
            f"{data['vendor']} {data['underlying']} day files under {data['root']} "
            f"(data.missing_close: {data['missing_close']})",
        ),
        (
            "stability_fit",
            float(len(fit_cfg.pillars)),
            f"{fit_text} (every other BreakEvenFitConfig setting at its library default, "
            "hashed with the config)",
        ),
        ("n_particles", float(cfg.n_particles), f"horizon {cfg.horizon:g}y"),
        (
            "n_paths",
            float(cfg.n_paths),
            f"seed {cfg.section('pricing')['seed']} on every date (CRN)",
        ),
        ("ssr_target", float(m["ssr_target"]), f"marking fit {m.get('fit', 'm7')!r}, stage 3 off"),
        ("skew_eps", float(m["skew_eps"]), ""),
        (
            "pillars",
            float(len(a["pillars"])),
            f"mode {a['mode']}, detail {a['detail']}, trades {a['trades']}",
        ),
        ("n_trades", float(rows["trade_id"].nunique()) if len(rows) else 0.0, cfg.rolling_every),
        ("fast", float(mode == "fast"), cfg.name),
        ("ssr_window", float(cfg.section("ssr")["window"]), ""),
        ("variance_trades", float(len(variance)), variance_strike_rule(variance)),
        ("rolling_daily", float(cfg.rolling_mark == "daily"), cfg.rolling_mark),
    ]
    for col, value, note in items:
        b.add_exact("setup", "backtest", col, value, unit="", source=SRC, note=note)


def variance_strike_rule(trades: Sequence[TradeSpec]) -> str:
    """How the book's variance products are struck (the narrative's sentence)."""
    if not trades:
        return "no variance product in the book"
    rules = sorted(
        {
            (
                "the inception surface's log-contract strike"
                if not isinstance(t.strike, (int, float))
                else f"a {100.0 * float(t.strike):g}% strike vol"
            )
            + ("" if t.strike_ratio is None else f" times {t.strike_ratio:g}")
            for t in trades
        }
    )
    return (
        f"struck at {' or '.join(rules)} with variance notional 1/(2 K_vol) (vega notional 1, "
        "values in vol points x100); the M8b studies used strike 0 in variance units instead"
    )


def _trade_results(b: ResultsBuilder, rows: pd.DataFrame) -> None:
    if rows.empty:
        b.add_exact("trades", "none", "n_rows", 0.0, unit="", source=SRC, note="no rows")
        return
    for tid, g in rows.groupby("trade_id", sort=False):
        g = g.sort_values("date")
        first, last = g.iloc[0], g.iloc[-1]
        unit, scale = str(first["unit"]), float(first["scale"])
        axes = {"fixed": float(first["book"] == "fixed")}
        note = (
            f"{first['book']} {first['kind']} from {first['inception']} in {first['unit']}; "
            f"{last['status']} {last['reason']}"
        ).strip()
        b.add_exact(
            "trades",
            str(tid),
            "maturity",
            float(first["maturity"]),
            unit="",
            source=SRC,
            note=note,
            axes=axes,
        )
        b.add_exact(
            "trades", str(tid), "strike", float(first["strike"]), unit="", source=SRC, axes=axes
        )
        b.add(
            "trades",
            str(tid),
            "first_value",
            scale * float(first["value"]),
            scale * float(first["value_stderr"]),
            unit=unit,
            source=SRC,
            axes=axes,
        )
        b.add(
            "trades",
            str(tid),
            "last_value",
            scale * float(last["value"]),
            scale * float(last["value_stderr"]),
            unit=unit,
            source=SRC,
            axes=axes,
        )
        b.add_exact("trades", str(tid), "n_dates", float(len(g)), unit="", source=SRC, axes=axes)
        if first["status"] == "inception":
            label = f"{tid} ({first['kind']}@{float(first['maturity']):g}y)"
            b.add(
                "inception",
                label,
                "value",
                scale * float(first["value"]),
                scale * float(first["value_stderr"]),
                unit=unit,
                source=SRC,
                note=str(tid),
                axes={"fixed": float(first["book"] == "fixed")},
            )


#: The stored rows keep the holder's sign, ``V(d) − V(d−1) + flows``; every study table, figure
#: and sentence presents the **desk** P&L, the desk being short the book (the M7 / M8b
#: convention): desk P&L = ``DESK_SIGN`` × holder P&L.
DESK_SIGN: Final[float] = -1.0
DESK_CONVENTION = (
    "desk P&L = -(V(d) - V(d-1) + cash flows paid on d): the desk is short the book (the M7 / "
    "M8b convention); the stored per-date rows keep the holder's sign"
)


def _window_results(
    b: ResultsBuilder,
    cfg: BacktestConfig,
    marked: Sequence[str],
    fits: Mapping[str, Mapping[str, Any]],
) -> None:
    """The window as the data shows it: dates, first and last close, return and close-to-close
    realised vol (252-day annualisation) — the narrative's description, computed."""
    spots = [float(fits[d]["close"]) for d in marked if "close" in fits[d]]
    if len(spots) < 2:
        b.add_exact("window", "spx", "n_closes", float(len(spots)), unit="", source=SRC)
        return
    r = np.diff(np.log(np.asarray(spots)))
    row = cfg.underlying
    note = f"{marked[0]} to {marked[-1]}"
    b.add_exact("window", row, "n_closes", float(len(spots)), unit="", source=SRC, note=note)
    b.add_exact("window", row, "first_close", spots[0], unit="", source=SRC)
    b.add_exact("window", row, "last_close", spots[-1], unit="", source=SRC)
    b.add_exact(
        "window", row, "return_pct", 100.0 * (spots[-1] / spots[0] - 1.0), unit="", source=SRC
    )
    b.add_exact(
        "window",
        row,
        "realised_vol_pct",
        100.0 * math.sqrt(TRADING_DAYS_PER_YEAR * float(np.mean(r * r))),
        unit="",
        source=SRC,
        note="close-to-close, 252-day annualisation, zero-mean",
    )


def _missing_pnl_results(
    b: ResultsBuilder,
    dates: Sequence[str],
    dones: Mapping[str, Mapping[str, Any]],
    rows: pd.DataFrame,
    live_counts: Mapping[str, int] | None = None,
) -> None:
    """Every date whose P&L is not complete, with the count of rows without a P&L (a failed
    date has no rows: the count is its live trades, ``live_counts``) and the reason and command
    stage 1 recorded — reported, never dropped."""
    lacking: dict[str, int] = {}
    if len(rows):
        mask = rows["status"].eq("unpriced") | (
            rows["status"].isin(["live", "settled", "paid"]) & rows["pnl"].isna()
        )
        lacking = {str(d): int(n) for d, n in rows[mask].groupby("date").size().items()}
    n_dates = 0
    for d in dates:
        done = dones[d]
        status = str(done.get("status"))
        if status not in ("incomplete", "failed") and not lacking.get(d):
            continue
        reasons = done.get("incomplete_reasons") or [done.get("error", "")]
        note = f"{status}: {'; '.join(str(x) for x in reasons if x)}"
        if done.get("command"):
            note += f" | recompute: {done['command']}"
        n_lacking = (live_counts or {}).get(d, 0) if status == "failed" else lacking.get(d, 0)
        b.add_exact(
            "pnl_missing",
            d,
            "rows_without_pnl",
            float(n_lacking),
            unit="",
            source=SRC,
            note=note[:900],
        )
        b.add_exact("pnl_missing", d, "failed", float(status == "failed"), unit="", source=SRC)
        n_dates += 1
    if n_dates == 0:
        b.add_exact(
            "pnl_missing",
            "none",
            "rows_without_pnl",
            0.0,
            unit="",
            source=SRC,
            note="every stored date carries its P&L",
        )
        b.add_exact("pnl_missing", "none", "failed", 0.0, unit="", source=SRC)


def _desk(holder_pnl: float) -> float:
    """A holder P&L as the desk P&L x100 (:data:`DESK_SIGN`), without a negative zero."""
    return 100.0 * DESK_SIGN * float(holder_pnl) + 0.0


#: What a stage-2 standard error is (:func:`aggregate` puts one of these in every note).
SE_PAIRED_CUMULATIVE = (
    "stderr: paired stderr of the cumulative P&L V(last) - V(inception) + cash flows"
)
SE_RSS_DATES = (
    "stderr: root sum of squares of correlated daily errors - not the error of the cumulative P&L"
)
SE_RSS_TRADES = (
    "stderr: root sum of squares of the date's per-trade errors, which share one seed - not "
    "the error of the sum"
)
SE_RSS_BUCKETS = (
    "stderr: root sum of squares of correlated bucket errors (a store without paired group "
    "errors) - indicative"
)
SE_RSS_PILLARS = (
    "stderr of a ladder bucket: root sum of squares of correlated pillar errors (a store "
    "without paired group errors) - indicative"
)
#: Buckets whose stored stderr was a root sum of squares over pillars before paired groups.
LADDER_BUCKETS: tuple[str, ...] = ("surface.vega_T", "surface.skew_T", "surface.curvature_T")
#: The buckets of each paired group (:data:`PAIRED_GROUPS`).
PAIRED_GROUP_BUCKETS: dict[str, frozenset[str]] = {
    "spot": frozenset({"spot.delta", "spot.gamma"}),
    "rates": frozenset({"rates.rho", "rates.repo"}),
    "surface": frozenset(
        {"surface.parallel_vega", "surface.vega_T", "surface.skew_T", "surface.curvature_T"}
    ),
    "params": frozenset(f"params.{p}" for p in PARAMS),
    "time": frozenset({"time.decay", "time.carry"}),
    "greeks": frozenset(GREEK_BUCKETS),
    "explained": frozenset({*GREEK_BUCKETS, "recalibration"}),
}


@dataclass(frozen=True)
class Aggregate:
    """A sum of stored P&L numbers, its standard error and what that error is (``note``:
    ``""`` for a row's own paired error)."""

    value: float
    stderr: float
    note: str = ""


def _cell_float(r: Mapping[str, Any], col: str) -> float:
    x = r.get(col, np.nan)
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _row_buckets(r: Mapping[str, Any], names: Sequence[str]) -> Aggregate:
    """A sum of buckets of ONE row: one bucket's own error; a stored paired group's error when
    the buckets are that group up to buckets that are zero on the row; the root sum of squares
    otherwise (labelled)."""
    value = float(np.nansum([_cell_float(r, bucket_column(n)) for n in names]))
    ses = {n: _cell_float(r, bucket_column(n) + "_stderr") for n in names}
    nonzero = [n for n, se in ses.items() if math.isfinite(se) and se != 0.0]
    old_ladder = [
        n
        for n in nonzero
        if n in LADDER_BUCKETS and not math.isfinite(_cell_float(r, group_column("surface")))
    ]
    note = SE_RSS_PILLARS if old_ladder else ""
    if len(nonzero) <= 1:
        return Aggregate(value, ses[nonzero[0]] if nonzero else 0.0, note)
    wanted = set(names)
    for g, members in PAIRED_GROUP_BUCKETS.items():
        se = _cell_float(r, group_column(g))
        if not math.isfinite(se):
            continue
        differ = (members - wanted) | (wanted - members)
        if all(
            _cell_float(r, bucket_column(n)) in (0.0,)
            or not math.isfinite(_cell_float(r, bucket_column(n)))
            for n in differ
        ):
            return Aggregate(value, se, note)
    rss = math.sqrt(sum(ses[n] ** 2 for n in nonzero))
    return Aggregate(value, rss, SE_RSS_BUCKETS)


def aggregate(g: pd.DataFrame, what: str | Sequence[str]) -> Aggregate:
    """**The one stage-2 aggregation** of stored P&L rows (P1): the sum over the rows of ``g``
    of the daily P&L (``what="pnl"``) or of the buckets ``what``, with its standard error and a
    note saying what that error is — every table, figure and sentence of the study uses it.

    * one row: the row's own paired error; a sum of buckets takes the row's paired group error
      (:data:`PAIRED_GROUPS`, stored from round 5 on) when the buckets are that group, else the
      root sum of squares, labelled :data:`SE_RSS_BUCKETS`;
    * one trade's P&L over its dates: the direct paired error of its cumulative P&L (the last
      row's ``cum_pnl_stderr``) when the rows are every P&L date since its inception and the
      last one carries it; otherwise the root sum of squares of the daily errors, labelled
      :data:`SE_RSS_DATES`;
    * buckets over dates, or several trades: the root sum of squares, labelled
      :data:`SE_RSS_DATES` (several dates) or :data:`SE_RSS_TRADES` (one date) — the dates and
      the trades share one seed, and no pricing-free estimator gives their covariance."""
    if g.empty:
        return Aggregate(0.0, 0.0)
    names = ["pnl"] if isinstance(what, str) else list(what)
    records = [{str(k): v for k, v in rec.items()} for rec in g.to_dict("records")]
    if len(records) == 1:
        r = records[0]
        if isinstance(what, str):
            return Aggregate(_cell_float(r, "pnl"), _cell_float(r, "pnl_stderr"))
        return _row_buckets(r, names)
    if isinstance(what, str):
        value = float(np.nansum(g["pnl"].to_numpy(dtype=float)))
        if g["trade_id"].nunique() == 1:
            ordered = g.sort_values("date")
            ages = [int(a) for a in ordered["age"].to_numpy(dtype=float)]
            last = (
                float(ordered["cum_pnl_stderr"].iloc[-1])
                if "cum_pnl_stderr" in ordered
                else float("nan")
            )
            if ages == list(range(1, len(ages) + 1)) and math.isfinite(last):
                return Aggregate(value, last, SE_PAIRED_CUMULATIVE)
        se = math.sqrt(float(np.nansum(g["pnl_stderr"].to_numpy(dtype=float) ** 2)))
    pillars = False
    if not isinstance(what, str):
        parts = [_row_buckets(r, names) for r in records]
        value = float(sum(p.value for p in parts))
        se = math.sqrt(sum(p.stderr**2 for p in parts if math.isfinite(p.stderr)))
        pillars = any(p.note == SE_RSS_PILLARS for p in parts)
    label = SE_RSS_DATES if g["date"].nunique() > 1 else SE_RSS_TRADES
    return Aggregate(value, se, f"{label}; {SE_RSS_PILLARS}" if pillars else label)


def unit_slug(unit: str) -> str:
    """A table-name part for a trade unit."""
    known = {
        "% notional": "pct_notional",
        "% of inception spot": "pct_spot",
        "vol pts (vega notional 1)": "vol_pts",
    }
    return known.get(unit) or re.sub(r"[^a-z0-9]+", "_", unit.lower()).strip("_") or "unit"


def book_row(unit: str) -> str:
    """The row of the fixed trades quoted in ``unit``: a book total never adds units (P2)."""
    return f"book (fixed, {unit})"


def _add(
    b: ResultsBuilder,
    table: str,
    row: str,
    column: str,
    a: Aggregate,
    unit: str,
    *,
    axes: Mapping[str, Any] | None = None,
    note: str = "",
) -> None:
    """A desk-sign aggregate (x100 of ``unit``) with its error's note."""
    b.add(
        table,
        row,
        column,
        _desk(a.value),
        100.0 * a.stderr,
        unit=unit,
        source=SRC,
        note="; ".join(x for x in (note, a.note) if x),
        axes=axes,
    )


def _pnl_results(b: ResultsBuilder, rows: pd.DataFrame) -> None:
    """Desk-sign P&L tables and series (:data:`DESK_SIGN`), x100 in each trade's own unit (its
    rows' ``unit``); a book total adds only the fixed trades quoted in one unit (P2); every
    standard error comes from :func:`aggregate`, whose note says what it is (P1)."""
    pnl = rows[rows["pnl"].notna()] if len(rows) else rows
    if pnl.empty:
        b.add_exact("pnl_trade", "none", "n_days", 0.0, unit="", source=SRC, note="no P&L rows yet")
        return
    fixed = pnl[pnl["book"] == "fixed"]
    trades = [(str(t), str(g["unit"].iloc[0]), g) for t, g in pnl.groupby("trade_id", sort=False)]
    books = [
        (book_row(str(u)), str(u), g)
        for u, g in fixed.groupby("unit", sort=False)
        if g["trade_id"].nunique() > 1
    ]
    for tid, unit, g in trades + books:
        axes = {"fixed": float(tid.startswith(("fixed:", "book (")))}
        members = (
            "" if tid in {t for t, _, _ in trades} else ", ".join(sorted(g["trade_id"].unique()))
        )
        _add(b, "pnl_trade", tid, "pnl", aggregate(g, "pnl"), unit, axes=axes, note=members)
        att = g[g["pnl_method"] == "attributed"]
        if att.empty:  # nothing was attributed: a missing number, not 0 +/- 0
            b.add(
                "pnl_trade",
                tid,
                "explained",
                float("nan"),
                float("nan"),
                unit=unit,
                source=SRC,
                axes=axes,
                note="not attributed (attribution.trades): its P&L is in 'paired, not attributed'",
            )
        else:
            _add(b, "pnl_trade", tid, "explained", aggregate(att, GREEK_BUCKETS), unit, axes=axes)
        for gname in ("recalibration", "residual", "cash_flows", "settlement", "unattributed"):
            _add(
                b,
                "pnl_trade",
                tid,
                gname,
                aggregate(g, dict(BUCKET_GROUPS)[gname]),
                unit,
                axes=axes,
            )
        b.add_exact(
            "pnl_trade", tid, "n_attributed", float(len(att)), unit="", source=SRC, axes=axes
        )
        b.add_exact("pnl_trade", tid, "n_days", float(len(g)), unit="", source=SRC, axes=axes)
    for tid, unit, g in trades + books:
        if tid.startswith(("fixed:", "book (")):
            for gname, names in BUCKET_GROUPS:
                _add(b, "pnl_bucket", gname, tid, aggregate(g, names), unit)
    if fixed.empty:
        return
    fixed = fixed.assign(month=fixed["date"].str[:7])
    for u, gu in fixed.groupby("unit", sort=False):
        for month, g in gu.groupby("month", sort=True):
            row = f"{month} ({u})"
            parts = 0.0
            for gname, names in BUCKET_GROUPS:
                a = aggregate(g, names)
                parts += a.value
                _add(b, "pnl_month", row, gname, a, str(u))
            tot = aggregate(g, "pnl")
            _add(b, "pnl_month", row, "total", tot, str(u))
            b.add_exact(
                "pnl_month",
                row,
                "check",
                _desk(tot.value - parts),
                unit="",
                source=SRC,
                note="total minus the sum of every bucket (0 up to rounding)",
            )
    dates = sorted(fixed["date"].unique())
    pos_book = {d: float(i) for i, d in enumerate(dates)}
    for (bu, d), g in fixed.groupby(["unit", "date"], sort=True):
        ax = {"t": pos_book[str(d)]}
        row = f"{d}|{bu}"
        _add(b, "daily_book", row, "pnl", aggregate(g, "pnl"), str(bu), axes=ax)
        for gname, names in BUCKET_GROUPS:
            _add(b, "daily_book", row, gname, aggregate(g, names), str(bu), axes=ax)
    all_dates = sorted(rows["date"].unique())
    pos = {d: float(i) for i, d in enumerate(all_dates)}
    for rec in fixed.to_dict("records"):
        r = {str(k): v for k, v in rec.items()}
        label = f"{r['date']}|{r['trade_id']}"
        ax = {"t": pos[str(r["date"])]}
        unit = str(r["unit"])
        one = pd.DataFrame([r])
        _add(b, "daily_trade", label, "pnl", aggregate(one, "pnl"), unit, axes=ax)
        if r["pnl_method"] == "attributed":
            _add(b, "daily_trade", label, "explained", aggregate(one, GREEK_BUCKETS), unit, axes=ax)
        cum, cum_se = _cell_float(r, "cum_pnl"), _cell_float(r, "cum_pnl_stderr")
        if math.isfinite(cum) and math.isfinite(cum_se):
            _add(
                b,
                "daily_trade",
                label,
                "cum_pnl",
                Aggregate(cum, cum_se, SE_PAIRED_CUMULATIVE),
                unit,
                axes=ax,
            )


def _fit_results(
    b: ResultsBuilder, cfg: BacktestConfig, ok: Sequence[str], fits: Mapping[str, Mapping[str, Any]]
) -> None:
    status_counts: dict[str, int] = {}
    frame_rows = []
    for i, d in enumerate(ok):
        f = fits[d]
        status_counts[str(f.get("status"))] = status_counts.get(str(f.get("status")), 0) + 1
        params = f.get("params", {})
        be = f.get("breakeven", {})
        note = f"{f.get('status')}; {'; '.join(f.get('messages', []))[:180]}"
        for k in PARAMS:
            b.add_exact(
                "params_daily",
                d,
                k,
                float(params.get(k, np.nan)),
                unit="",
                source=SRC,
                note=note,
                axes={"t": float(i)},
            )
        for k in PARAM_COLUMNS:
            b.add_exact(
                "params_daily",
                d,
                f"be_{k}",
                float(be.get(k, np.nan)),
                unit="",
                source=SRC,
                axes={"t": float(i)},
            )
        se = f.get("se", {})
        frame_rows.append(
            {
                "date": d,
                **{k: float(be.get(k, np.nan)) for k in PARAM_COLUMNS},
                **{f"{k}_se": np.nan if se.get(k) is None else float(se[k]) for k in PARAM_COLUMNS},
            }
        )
    for s, n in sorted(status_counts.items()):
        b.add_exact("fits", s, "n_dates", float(n), unit="", source=SRC)
    if frame_rows:
        frame = pd.DataFrame(frame_rows)
        _param_summary(b, frame, ok, fits)
        st = cfg.section("stability")
        flags = _flags(frame, share=float(st["share"]), band=float(st["band"]))
        _flag_rows(
            b,
            "params_flags",
            flags,
            "marking fits: no standard error by construction "
            "(the marking targets carry no sampling error)",
        )
        _stability_hist(b, cfg, ok, fits)


def _flags(frame: pd.DataFrame, *, share: float, band: float) -> pd.DataFrame:
    """:func:`~volsto.calibration.stability.flag_unidentified`, without numpy's "All-NaN slice"
    warning: the marking fits carry no standard error by construction, so a one-date window's
    median standard error is NaN — the expected, reported outcome."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        return flag_unidentified(frame, PARAM_COLUMNS, share=share, band=band)


def _param_summary(
    b: ResultsBuilder, frame: pd.DataFrame, ok: Sequence[str], fits: Mapping[str, Mapping[str, Any]]
) -> None:
    book = pd.DataFrame(
        [{k: float(fits[d].get("params", {}).get(k, np.nan)) for k in PARAMS} for d in ok]
    )
    for k in PARAMS:
        x = book[k].to_numpy(dtype=float)
        dx = np.abs(np.diff(x)) if x.size > 1 else np.array([np.nan])
        for col, v in (
            ("first", x[0]),
            ("last", x[-1]),
            ("min", np.nanmin(x)),
            ("max", np.nanmax(x)),
            ("median_abs_change", np.nanmedian(dx) if np.isfinite(dx).any() else np.nan),
        ):
            b.add_exact("params_summary", k, col, float(v), unit="", source=SRC)


def _flag_rows(b: ResultsBuilder, table: str, flags: pd.DataFrame, note: str) -> None:
    for rec in flags.to_dict("records"):
        p = str(rec["param"])
        for col in (
            "share_beyond_se",
            "median_abs_change",
            "median_se",
            "n_without_se",
            "n_changes",
        ):
            b.add_exact(
                table,
                p,
                col,
                float(rec[col]),
                unit="",
                source=SRC,
                note=note if col == "share_beyond_se" else "",
            )
        share = float(rec["share_beyond_se"])
        b.add_exact(
            table,
            p,
            "unidentified",
            float(bool(rec["unidentified"])) if math.isfinite(share) else float("nan"),
            unit="",
            source=SRC,
            note="" if math.isfinite(share) else "no usable standard error: cannot flag",
        )


def _history(ok: Sequence[str], fits: Mapping[str, Mapping[str, Any]]) -> SurfaceHistory | None:
    if len(ok) < 2:
        return None
    h0 = fits[ok[0]]["history"]
    return SurfaceHistory.from_arrays(
        list(ok),
        h0["T"],
        [fits[d]["history"]["vs_vol"] for d in ok],
        [fits[d]["history"]["atm_vol"] for d in ok],
        [fits[d]["history"]["skew"] for d in ok],
        [fits[d]["history"]["ln_spot"] for d in ok],
    )


def _stability_hist(
    b: ResultsBuilder, cfg: BacktestConfig, ok: Sequence[str], fits: Mapping[str, Mapping[str, Any]]
) -> None:
    st = cfg.section("stability")
    wv, ws = int(st["window_vol"]), int(st["window_ssr"])
    need = max(wv, ws) + 2
    hist = _history(ok, fits)
    if hist is None or hist.n_dates < need:
        reason = (
            f"the historical-mode rolling fit (SPEC §15 Part 4) needs {need} dates (windows "
            f"{wv}/{ws} plus two fitted dates); {len(ok)} available"
        )
        for p in PARAM_COLUMNS:
            b.add(
                "stability_hist",
                p,
                "share_beyond_se",
                float("nan"),
                float("nan"),
                unit=DIMENSIONLESS,
                source=SRC,
                note=reason,
            )
        return
    frame = rolling_fit(
        hist,
        cfg.stability_fit_config(),
        window_vol=wv,
        window_ssr=ws,
        start=hist.dates[max(wv, ws)],
    )
    flags = _flags(frame, share=float(st["share"]), band=float(st["band"]))
    _flag_rows(
        b,
        "stability_hist",
        flags,
        f"historical-mode rolling fit, windows {wv}/{ws}, {len(frame)} dates",
    )
    for i, rec in enumerate(frame.to_dict("records")):
        d = str(pd.Timestamp(rec["date"]).date())
        for p in PARAM_COLUMNS:
            se = rec[f"{p}_se"]
            if np.isfinite(se):
                b.add(
                    "stability_daily",
                    d,
                    p,
                    float(rec[p]),
                    float(se),
                    unit=DIMENSIONLESS,
                    source=SRC,
                    axes={"t": float(i)},
                )
            else:
                b.add_exact(
                    "stability_daily",
                    d,
                    p,
                    float(rec[p]),
                    unit="",
                    source=SRC,
                    note=f"no finite standard error (raw {rec[f'{p}_se_raw']})",
                    axes={"t": float(i)},
                )


def _ssr_results(
    b: ResultsBuilder, cfg: BacktestConfig, ok: Sequence[str], fits: Mapping[str, Mapping[str, Any]]
) -> None:
    q = cfg.section("ssr")
    window = int(q["window"])
    pillars = [float(x) for x in q["pillars"]]
    hist = _history(ok, fits)
    target = float(cfg.section("marking")["ssr_target"])
    for T in pillars:
        realised: list[tuple[float, float]] = []
        first_order: list[float] = []
        for i, d in enumerate(ok):
            f = fits[d]
            label = f"{d}|{T:g}"
            ax = {"t": float(i), "T": T}
            fo = _pillar_value(f.get("pillars", []), f.get("ssr_first_order", []), T)
            first_order.append(fo)
            b.add_exact("ssr_daily", label, "target", target, unit="", source=SRC, axes=ax)
            b.add_exact(
                "ssr_daily",
                label,
                "first_order",
                fo,
                unit="",
                source=SRC,
                note="" if np.isfinite(fo) else f"{T:g}y is not a fitted pillar",
                axes=ax,
            )
            if hist is None or i < window:
                b.add(
                    "ssr_daily",
                    label,
                    "realised",
                    float("nan"),
                    float("nan"),
                    unit=DIMENSIONLESS,
                    source=SRC,
                    note=f"needs {window + 1} dates ({window} increments); {i + 1} available",
                    axes=ax,
                )
                continue
            e = hist.ssr_hist(T, window, end=pd.Timestamp(d))
            realised.append((e.ssr, e.se))
            b.add(
                "ssr_daily",
                label,
                "realised",
                float(e.ssr),
                float(e.se),
                unit=DIMENSIONLESS,
                source=SRC,
                note=f"window {window}, R2 {e.r2:.2f}",
                axes=ax,
            )
        row = f"{T:g}y"
        b.add_exact("ssr", row, "target", target, unit="", source=SRC)
        fo_arr = np.asarray(first_order, dtype=float)
        b.add_exact(
            "ssr",
            row,
            "first_order_median",
            float(np.nanmedian(fo_arr)) if np.isfinite(fo_arr).any() else float("nan"),
            unit="",
            source=SRC,
        )
        if realised:
            b.add(
                "ssr",
                row,
                "realised_last",
                realised[-1][0],
                realised[-1][1],
                unit=DIMENSIONLESS,
                source=SRC,
            )
            b.add_exact(
                "ssr",
                row,
                "realised_median",
                float(np.median([r[0] for r in realised])),
                unit="",
                source=SRC,
            )
        else:
            reason = f"fewer than {window + 1} dates: no realised SSR"
            b.add(
                "ssr",
                row,
                "realised_last",
                float("nan"),
                float("nan"),
                unit=DIMENSIONLESS,
                source=SRC,
                note=reason,
            )
            b.add_exact(
                "ssr", row, "realised_median", float("nan"), unit="", source=SRC, note=reason
            )
        b.add_exact("ssr", row, "n_windows", float(len(realised)), unit="", source=SRC)


def _pillar_value(ts: Sequence[float], values: Sequence[float], T: float) -> float:
    for t, v in zip(ts, values):
        if abs(float(t) - T) < 1e-9:
            return float(v)
    return float("nan")


def _vko_results(b: ResultsBuilder, rows: pd.DataFrame) -> None:
    vko = rows[rows["kind"] == "vko_put"] if len(rows) else rows
    if vko.empty:
        b.add_exact(
            "vko", "none", "n_dates", 0.0, unit="", source=SRC, note="no VKO trade in the book"
        )
        return
    all_dates = sorted(rows["date"].unique())
    pos = {d: float(i) for i, d in enumerate(all_dates)}
    for tid, g in vko.groupby("trade_id", sort=False):
        g = g.sort_values("date")
        first, last = g.iloc[0], g.iloc[-1]
        scale, unit = float(first["scale"]), str(first["unit"])
        remaining = float(first["maturity"]) - float(last["age"]) / TRADING_DAYS_PER_YEAR
        outcome = (
            "matured in the window"
            if last["status"] in ("settled", "paid")
            else (
                f"not observable in the window: {remaining:.2f}y of the "
                f"{float(first['maturity']):g}y life remain"
            )
        )
        b.add(
            "vko",
            str(tid),
            "inception_value",
            scale * float(first["value"]),
            scale * float(first["value_stderr"]),
            unit=unit,
            source=SRC,
            note=outcome,
        )
        b.add(
            "vko",
            str(tid),
            "last_value",
            scale * float(last["value"]),
            scale * float(last["value_stderr"]),
            unit=unit,
            source=SRC,
        )
        b.add_exact(
            "vko",
            str(tid),
            "realised_vol",
            100.0 * float(last["realised_vol"]),
            unit="vol pts",
            source=SRC,
        )
        b.add_exact(
            "vko", str(tid), "vol_ko", 100.0 * float(first["vol_ko"]), unit="vol pts", source=SRC
        )
        b.add_exact("vko", str(tid), "knocked_out", float(last["knocked_out"]), unit="", source=SRC)
        b.add_exact("vko", str(tid), "years_remaining", remaining, unit="", source=SRC)
        for _, r in g.iterrows():
            label = f"{r['date']}|{tid}"
            ax = {"t": pos[str(r["date"])]}
            b.add(
                "vko_daily",
                label,
                "value",
                scale * float(r["value"]),
                scale * float(r["value_stderr"]),
                unit=unit,
                source=SRC,
                note=str(tid),
                axes=ax,
            )
            b.add_exact(
                "vko_daily",
                label,
                "realised_vol",
                100.0 * float(r["realised_vol"]),
                unit="vol pts",
                source=SRC,
                note=str(tid),
                axes=ax,
            )
            b.add_exact(
                "vko_daily",
                label,
                "vol_ko",
                100.0 * float(r["vol_ko"]),
                unit="vol pts",
                source=SRC,
                note=str(tid),
                axes=ax,
            )


def _cost_results(
    b: ResultsBuilder,
    dates: Sequence[str],
    dones: Mapping[str, Mapping[str, Any]],
    fits: Mapping[str, Mapping[str, Any]],
) -> None:
    keys = ("import", "fit", "calibration", "calibration_previous", "pricing", "total")
    for k in keys:
        vals = [
            float(dones[d].get("timings", {}).get(k, 0.0) or 0.0)
            for d in dates
            if dones[d].get("status") == "ok"
        ]
        b.add_exact("cost", k, "total_s", float(np.sum(vals)) if vals else 0.0, unit="", source=SRC)
        b.add_exact(
            "cost",
            k,
            "median_s",
            float(np.median(vals)) if vals else float("nan"),
            unit="",
            source=SRC,
        )
    n_cal = sum(len(dones[d].get("calibrated") or []) for d in dates)
    b.add_exact(
        "cost",
        "calibrations",
        "total_s",
        float(n_cal),
        unit="",
        source=SRC,
        note="count of leverage calibrations recorded by stage 1",
    )
    b.add_exact("cost", "calibrations", "median_s", float("nan"), unit="", source=SRC)
    pricings = [float(dones[d].get("timings", {}).get("pricings", 0.0) or 0.0) for d in dates]
    b.add_exact(
        "cost",
        "pricings",
        "total_s",
        float(np.sum(pricings)),
        unit="",
        source=SRC,
        note="count of Monte Carlo pricings",
    )
    b.add_exact(
        "cost",
        "pricings",
        "median_s",
        float(np.median(pricings)) if pricings else float("nan"),
        unit="",
        source=SRC,
    )
    repaired = sum(1 for f in fits.values() if f.get("calendar", {}).get("calendar_repaired"))
    fallback = sum(1 for f in fits.values() if f.get("calendar", {}).get("calendar_fallback"))
    b.add_exact("fits", "calendar repaired", "n_dates", float(repaired), unit="", source=SRC)
    b.add_exact("fits", "calendar fallback", "n_dates", float(fallback), unit="", source=SRC)


# -- tables ----------------------------------------------------------------------------------


def _columns_or(results: Results, table: str, default: tuple[str, ...]) -> list[str]:
    """The columns of ``table``, or ``default`` when the table is absent (the spec is then
    dropped by :func:`tables`)."""
    return results.columns(table) if table in results.tables() else list(default)


def _units(results: Results, table: str, column: str) -> dict[str, list[str]]:
    """``{unit: rows}`` of one column of a results table (row order kept)."""
    if table not in results.tables() or column not in results.columns(table):
        return {}
    long = results.long(table, column)
    out: dict[str, list[str]] = {}
    for row, unit in zip(long["row"], long["unit"]):
        out.setdefault(str(unit), []).append(str(row))
    return out


def stderr_caption(results: Results, table: str, rows: Sequence[str]) -> str:
    """What the standard errors of ``rows`` of ``table`` are (from :func:`aggregate`'s notes)."""
    long = results.long(table)
    long = long[long["row"].isin(list(rows)) & ~long["exact"].astype(bool)]
    kinds: dict[str, set[str]] = {}
    for col, note in zip(long["column"], long["note"]):
        for kind in (
            SE_PAIRED_CUMULATIVE,
            SE_RSS_DATES,
            SE_RSS_TRADES,
            SE_RSS_BUCKETS,
            SE_RSS_PILLARS,
        ):
            if kind in str(note):
                kinds.setdefault(kind.removeprefix("stderr: "), set()).add(str(col))
    if not kinds:
        return "Standard errors: each number's own paired one."
    parts = [f"{k} ({', '.join(sorted(cols))})" for k, cols in kinds.items()]
    return "Standard errors: " + "; ".join(parts) + "."


def pnl_trade_specs(results: Results) -> list[TableSpec]:
    """The cumulative P&L tables, one per unit (a column never mixes units)."""
    specs = []
    for unit, rows in _units(results, "pnl_trade", "pnl").items():
        specs.append(
            TableSpec(
                f"pnl_trade_{unit_slug(unit)}",
                f"Cumulative desk P&L (the desk is short the book), x100 in {unit}: per trade "
                "and, for fixed trades quoted in this unit, their book; the Greek-explained part, "
                "the recalibration bucket, the residual, cash flows, settlements and the paired "
                "P&L of the trades not attributed. " + stderr_caption(results, "pnl_trade", rows),
                "pnl_trade",
                tuple(
                    Column(key, header, unit=unit if key != "n_attributed" else "", digits=d)
                    for key, header, d in (
                        ("pnl", "desk P&L", 4),
                        ("explained", "explained", 4),
                        ("recalibration", "recalibration", 4),
                        ("residual", "residual", 4),
                        ("cash_flows", "cash flows", 4),
                        ("settlement", "settlement", 4),
                        ("unattributed", "paired, not attributed", 4),
                        ("n_attributed", "attributed days", 3),
                    )
                ),
                rows=tuple(rows),
                row_header="trade",
            )
        )
    return specs


def tables(results: Results) -> list[TableSpec]:
    specs = []
    for unit, rows in _units(results, "trades", "first_value").items():
        specs.append(
            TableSpec(
                f"trades_{unit_slug(unit)}",
                f"The book, trades quoted in {unit}: first and last marked values (holder's "
                "value, Monte Carlo, 1 stderr), x100.",
                "trades",
                (
                    Column("maturity", "maturity [y]", digits=3),
                    Column("strike", "strike", digits=5),
                    Column("first_value", "first value (holder)", unit=unit),
                    Column("last_value", "last value (holder)", unit=unit),
                    Column("n_dates", "dates", digits=3),
                ),
                rows=tuple(rows),
                row_header="trade",
            )
        )
    if not specs and "trades" in results.tables():
        specs.append(
            TableSpec(
                "trades", "The book: no rows yet.", "trades", (Column("n_rows", "rows", digits=3),)
            )
        )
    for unit, rows in _units(results, "inception", "value").items():
        specs.append(
            TableSpec(
                f"inception_{unit_slug(unit)}",
                f"Inception prices of the fixed and rolling trades quoted in {unit} (the rolling "
                "book strikes on the first date of each month), x100.",
                "inception",
                (Column("value", "value at inception (holder)", unit=unit),),
                rows=tuple(rows),
                row_header="trade",
            )
        )
    if "pnl" not in _columns_or(results, "pnl_trade", ("pnl",)):
        specs.append(
            TableSpec(
                "pnl_trade",
                "Cumulative desk P&L: no stored date carries a P&L yet (see the dates without "
                "P&L).",
                "pnl_trade",
                (Column("n_days", "days with a P&L", digits=3),),
                row_header="trade",
            )
        )
    else:
        specs += pnl_trade_specs(results)
    if "pnl_bucket" in results.tables():
        cols = tuple(Column(c, c.replace("fixed:", "")) for c in results.columns("pnl_bucket"))
        specs.append(
            TableSpec(
                "pnl_bucket",
                "Cumulative desk P&L by attribution bucket (x100): fixed trades and the books of "
                "fixed trades sharing a unit; each column in its trade's (or book's) unit. "
                + stderr_caption(results, "pnl_bucket", results.rows("pnl_bucket")),
                "pnl_bucket",
                cols,
                row_header="bucket",
            )
        )
    keep = (*(g for g, _ in BUCKET_GROUPS), "total", "check")
    for unit, rows in _units(results, "pnl_month", "total").items():
        specs.append(
            TableSpec(
                f"pnl_month_{unit_slug(unit)}",
                f"Desk P&L per month by bucket of the fixed trades quoted in {unit} (x100); every "
                "bucket is shown, so they add up to the total ('check' is the difference, 0 up to "
                "rounding). " + stderr_caption(results, "pnl_month", rows),
                "pnl_month",
                tuple(
                    Column(
                        c,
                        c.replace("_", " "),
                        unit="" if c == "check" else unit,
                        digits=3 if c == "check" else 4,
                    )
                    for c in keep
                ),
                rows=tuple(rows),
                row_header="month",
            )
        )
    if "pnl_missing" in results.tables():
        specs.append(
            TableSpec(
                "pnl_missing",
                "Dates whose P&L is incomplete or missing (their reasons and commands are in "
                "study.md).",
                "pnl_missing",
                (
                    Column("rows_without_pnl", "rows without P&L", digits=3),
                    Column("failed", "date failed", digits=1),
                ),
                row_header="date",
            )
        )
    if "window" in results.tables() and "first_close" in results.columns("window"):
        specs.append(
            TableSpec(
                "window",
                "The window as the data shows it (closes of the marked dates).",
                "window",
                (
                    Column("n_closes", "closes", digits=4),
                    Column("first_close", "first close", digits=6),
                    Column("last_close", "last close", digits=6),
                    Column("return_pct", "return [%]", digits=3),
                    Column("realised_vol_pct", "realised vol [%]", digits=3),
                ),
                row_header="underlying",
            )
        )
    specs += [
        TableSpec(
            "params_summary",
            "Marking-fit parameters over the window (exact: a fit is a deterministic function "
            "of the day's surface).",
            "params_summary",
            (
                Column("first", "first"),
                Column("last", "last"),
                Column("min", "min"),
                Column("max", "max"),
                Column("median_abs_change", "median daily change"),
            ),
            row_header="parameter",
        ),
        TableSpec(
            "params_flags",
            "Stability flags (SPEC 15 Part 4) on the marking fits' break-even parameters.",
            "params_flags",
            (
                Column("share_beyond_se", "share beyond se", digits=3),
                Column("median_abs_change", "median change"),
                Column("n_without_se", "dates without se", digits=3),
                Column("unidentified", "flag", digits=1),
            ),
            row_header="parameter",
        ),
        TableSpec(
            "stability_hist",
            "Stability flags of the historical-mode rolling fit (SPEC 15 Part 4).",
            "stability_hist",
            tuple(
                Column(c, c.replace("_", " "), digits=3)
                for c in _columns_or(results, "stability_hist", ("share_beyond_se",))
            ),
            row_header="parameter",
        ),
        TableSpec(
            "ssr",
            "Realised SSR (rolling window, Newey-West stderr) against the marked target and "
            "the fit's first-order SSR.",
            "ssr",
            (
                Column("target", "target", digits=3),
                Column("first_order_median", "fit, first order (median)", digits=3),
                Column("realised_last", "realised (last window)"),
                Column("realised_median", "realised (median)", digits=3),
                Column("n_windows", "windows", digits=3),
            ),
            row_header="pillar",
        ),
        TableSpec(
            "vko",
            "Vol knock-out puts: marked price (Monte Carlo) against the realised state.",
            "vko",
            tuple(
                Column(c, c.replace("_", " "), digits=4)
                for c in _columns_or(results, "vko", ("n_dates",))
            ),
            row_header="trade",
        ),
        TableSpec(
            "fits",
            "Marking-fit statuses and calendar repairs over the stored dates.",
            "fits",
            (Column("n_dates", "dates", digits=3),),
            row_header="status",
        ),
        TableSpec(
            "cost",
            "Stage-1 wall clock by step (seconds; counts for calibrations and pricings).",
            "cost",
            (Column("total_s", "total", digits=5), Column("median_s", "median per date", digits=3)),
            row_header="step",
        ),
    ]
    present = set(results.tables())
    return [s for s in specs if s.table in present]


# -- figures ---------------------------------------------------------------------------------


def _unit_panels(n: int) -> tuple[Figure, list[Any]]:
    """One panel per unit (at least one)."""
    fig, axes = style.new_figure(1, max(n, 1), width=4.6 * max(n, 1), height=3.8)
    return fig, list(np.atleast_1d(axes).ravel())


def _draw_book(results: Results) -> Figure:
    if "daily_book" not in results.tables():
        fig, ax = style.new_figure()
        ax.text(0.5, 0.5, "no P&L rows yet", ha="center", transform=ax.transAxes)
        return fig
    long = results.long("daily_book")
    units = list(dict.fromkeys(long["unit"]))
    fig, axes = _unit_panels(len(units))
    for ax, unit in zip(axes, units):
        df = long[long["unit"] == unit]
        wide = df.pivot(index="row", columns="column", values="value")
        errs = df.pivot(index="row", columns="column", values="stderr")
        order = df.drop_duplicates("row").sort_values("axis_t")["row"].tolist()
        wide, errs = wide.reindex(order), errs.reindex(order)
        t = np.arange(len(wide))
        for k, (label, groups) in enumerate(FIGURE_GROUPS, start=1):
            cols = [g for g in groups if g in wide]
            if not cols:
                continue
            series = np.nansum(np.column_stack([wide[c].to_numpy() for c in cols]), axis=1)
            ax.plot(t, np.cumsum(series), label=label, color=style.series_color(k))
        total = np.nancumsum(wide["pnl"].to_numpy())
        se = np.sqrt(np.nancumsum(errs["pnl"].to_numpy() ** 2))
        style.mc_errorbar(
            ax,
            t,
            total,
            se,
            label="desk P&L, cumulative (bars: RSS of daily errors)",
            series=0,
        )
        ax.set_xlabel("trading date index")
        _integer_ticks(ax)
        ax.set_ylabel(f"desk P&L x100 [{unit}]")
        ax.set_title(f"Fixed trades in {unit}")
        ax.legend(fontsize=6)
    return fig


#: The book figure's bucket groups (at most seven: the palette has eight hues).
FIGURE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("spot: delta + gamma", ("delta", "gamma")),
    ("surface: vega + skew + curvature", ("vega", "skew", "curvature")),
    ("model parameters", ("params",)),
    ("rates + time", ("rates", "time")),
    ("recalibration", ("recalibration",)),
    ("residual", ("residual",)),
    ("cash flows + settlements + paired", ("cash_flows", "settlement", "unattributed")),
)


def _integer_ticks(ax: Any) -> None:
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _trade_of(row: str) -> str:
    return row.split("|", 1)[1] if "|" in row else row


def _draw_trades(results: Results) -> Figure:
    if "daily_trade" not in results.tables():
        fig, ax = style.new_figure()
        ax.text(0.5, 0.5, "no P&L rows yet", ha="center", transform=ax.transAxes)
        return fig
    long = results.long("daily_trade")
    long = long.assign(trade=[_trade_of(str(r)) for r in long["row"]])
    units = list(dict.fromkeys(long["unit"]))
    fig, axes = _unit_panels(len(units))
    k = 0
    for ax, unit in zip(axes, units):
        for tid, g in long[long["unit"] == unit].groupby("trade", sort=False):
            daily = g[g["column"] == "pnl"].sort_values("axis_t")
            cum = g[g["column"] == "cum_pnl"].sort_values("axis_t")
            name = str(tid).replace("fixed:", "")
            if len(cum) == len(daily) and len(cum):
                # the stored cumulative P&L with its direct paired error
                x, y, e = (cum[c].to_numpy() for c in ("axis_t", "value", "stderr"))
                label = f"{name} (paired stderr)"
            else:
                x = daily["axis_t"].to_numpy()
                y = np.cumsum(daily["value"].to_numpy())
                e = np.sqrt(np.cumsum(daily["stderr"].to_numpy() ** 2))
                label = f"{name} (RSS of daily errors, not the cumulative error)"
            style.mc_errorbar(ax, x, y, e, series=k % len(style.PALETTE), label=label)
            k += 1
        ax.set_xlabel("trading date index")
        _integer_ticks(ax)
        ax.set_ylabel(f"cumulative desk P&L x100 [{unit}]")
        ax.set_title(f"Fixed trades in {unit} (1 stderr)")
        ax.legend(fontsize=6)
    return fig


def _draw_explained(results: Results) -> Figure:
    if "daily_trade" not in results.tables() or "explained" not in results.columns("daily_trade"):
        fig, ax = style.new_figure()
        ax.text(0.5, 0.5, "no attributed day yet", ha="center", transform=ax.transAxes)
        return fig
    long = results.long("daily_trade")
    units = list(dict.fromkeys(long.loc[long["column"] == "explained", "unit"]))
    fig, axes = _unit_panels(len(units))
    for ax, unit in zip(axes, units):
        df = long[long["unit"] == unit]
        val = df.pivot(index="row", columns="column", values="value")
        err = df.pivot(index="row", columns="column", values="stderr")
        ok = val["explained"].notna()
        x, y = val.loc[ok, "pnl"].to_numpy(), val.loc[ok, "explained"].to_numpy()
        ax.errorbar(
            x,
            y,
            xerr=err.loc[ok, "pnl"].to_numpy(),
            yerr=err.loc[ok, "explained"].to_numpy(),
            fmt="o",
            ms=3,
            color=style.series_color(0),
            label="daily desk P&L per trade (x, y: 1 stderr)",
        )
        lim = float(np.nanmax(np.abs(np.concatenate([x, y])))) if x.size else 1.0
        ax.plot(
            [-lim, lim],
            [-lim, lim],
            color=style.INK["spine"],
            linewidth=0.8,
            label="explained = actual",
        )
        ax.set_xlabel(f"actual daily desk P&L x100 [{unit}]")
        ax.set_ylabel(f"Greek-explained x100 [{unit}]")
        ax.set_title(f"Explained against actual ({unit})")
        ax.legend(fontsize=6)
    return fig


def _draw_params(results: Results) -> Figure:
    fig, axes = style.new_figure(2, 3, width=9.6, height=5.6)
    df = results.pivot("params_daily")
    t = np.arange(len(df))
    for ax, name in zip(np.ravel(axes), ("nu", "theta", "k1", "rho12", "rho_SX1", "rho_SX2")):
        style.mc_errorbar(
            ax,
            t,
            df[name].to_numpy(),
            df[f"{name}_stderr"].to_numpy(),
            exact=np.ones(len(t), dtype=bool),
            series=0,
        )
        ax.set_title(name)
        ax.set_xlabel("date index")
        _integer_ticks(ax)
    return fig


def _draw_ssr(results: Results) -> Figure:
    fig, ax = style.new_figure()
    long = results.long("ssr_daily")
    for k, (T, g) in enumerate(long.groupby("axis_T", sort=True)):
        if k >= 4:
            break
        r = g[g["column"] == "realised"].sort_values("axis_t")
        fo = g[g["column"] == "first_order"].sort_values("axis_t")
        style.mc_errorbar(
            ax,
            r["axis_t"].to_numpy(),
            r["value"].to_numpy(),
            r["stderr"].to_numpy(),
            series=k,
            label=f"realised {T:g}y (1 stderr)",
        )
        if np.isfinite(fo["value"].to_numpy()).any():
            ax.plot(
                fo["axis_t"].to_numpy(),
                fo["value"].to_numpy(),
                linestyle="--",
                color=style.series_color(k),
                label=f"fit first-order {T:g}y",
            )
    tgt = long[long["column"] == "target"]["value"]
    if len(tgt):
        ax.axhline(
            float(tgt.iloc[0]), color=style.INK["spine"], linewidth=0.8, label="marked target"
        )
    ax.set_xlabel("trading date index")
    _integer_ticks(ax)
    ax.set_ylabel("SSR")
    ax.set_title("Realised SSR (rolling window) against the mark")
    ax.legend(fontsize=7)
    return fig


def _draw_vko(results: Results) -> Figure:
    fig, axes = style.new_figure(1, 2, width=9.6, height=3.6)
    if "vko_daily" not in results.tables():
        for ax in axes:
            ax.text(0.5, 0.5, "no VKO trade", ha="center", transform=ax.transAxes)
        return fig
    for k, (tid, g) in enumerate(results.long("vko_daily").groupby("note", sort=False)):
        if k >= 4:
            break
        v = g[g["column"] == "value"].sort_values("axis_t")
        rv = g[g["column"] == "realised_vol"].sort_values("axis_t")
        ko = g[g["column"] == "vol_ko"]
        label = str(tid).replace("rolling:", "").replace("fixed:", "")
        style.mc_errorbar(
            axes[0],
            v["axis_t"].to_numpy(),
            v["value"].to_numpy(),
            v["stderr"].to_numpy(),
            series=k,
            label=label,
        )
        axes[1].plot(
            rv["axis_t"].to_numpy(),
            rv["value"].to_numpy(),
            color=style.series_color(k),
            label=f"{label} realised",
        )
        if len(ko):
            axes[1].axhline(
                float(ko["value"].iloc[0]),
                color=style.series_color(k),
                linestyle=":",
                linewidth=0.8,
            )
    axes[0].set_title("VKO marked price (1 stderr)")
    axes[0].set_xlabel("trading date index")
    _integer_ticks(axes[0])
    axes[1].set_title("realised vol to date vs vol_ko (dotted)")
    axes[1].set_xlabel("trading date index")
    _integer_ticks(axes[1])
    axes[0].legend(fontsize=6)
    axes[1].legend(fontsize=6)
    return fig


def figures(results: Results) -> list[FigureSpec]:
    specs = [
        FigureSpec(
            "book_pnl",
            "Fixed trades, one panel per unit: cumulative desk P&L and the cumulative "
            "attribution buckets, desk sign (bars: root sum of squares of correlated daily "
            "errors - not the error of the cumulative P&L).",
            _draw_book,
        ),
        FigureSpec(
            "trade_pnl",
            "Cumulative desk P&L of each fixed trade, one panel per unit (1 stderr bars: the "
            "stored paired stderr of the cumulative P&L where the legend says so, else the "
            "root sum of squares of correlated daily errors).",
            _draw_trades,
        ),
        FigureSpec(
            "explained",
            "Greek-explained against actual daily desk P&L of the fixed trades, one panel per "
            "unit (x bars: the paired P&L's stderr; y bars: the explained sum's).",
            _draw_explained,
        ),
        FigureSpec(
            "params", "Marking-fit parameters by date (exact values: no bars).", _draw_params
        ),
        FigureSpec(
            "ssr",
            "Realised SSR by pillar (1 stderr, Newey-West) against the marked target and the "
            "fit's first-order SSR.",
            _draw_ssr,
        ),
        FigureSpec(
            "vko",
            "VKO puts: marked price (1 stderr) and realised vol to date against the vol barrier.",
            _draw_vko,
        ),
    ]
    present = set(results.tables())
    need = {"params": "params_daily", "ssr": "ssr_daily"}
    return [s for s in specs if need.get(s.name, "setup") in present]


# -- narrative -------------------------------------------------------------------------------


def _v(results: Results, table: str, row: str, column: str) -> tuple[float, float]:
    try:
        return results.value(table, row, column)
    except KeyError:
        return float("nan"), float("nan")


def _note(results: Results, table: str, row: str, column: str) -> str:
    try:
        return str(results.record(table, row, column)["note"])
    except KeyError:
        return ""


def _fmt(v: float, se: float = float("nan"), digits: int = 3) -> str:
    if not math.isfinite(v):
        return "n/a"
    if math.isfinite(se):
        return f"{v:.{digits}f} +/- {se:.{digits}f}"
    return f"{v:.{digits}f}"


def _stability_sentence(results: Results) -> str:
    if "stability_hist" not in results.tables():
        return "No stability result."
    rows = results.rows("stability_hist")
    share = [_v(results, "stability_hist", r, "share_beyond_se")[0] for r in rows]
    note = _note(results, "stability_hist", rows[0], "share_beyond_se")
    if not any(math.isfinite(x) for x in share):
        return f"Historical-mode rolling fit: not run — {note}."
    flagged = [r for r in rows if _v(results, "stability_hist", r, "unidentified")[0] >= 1]
    return (
        f"Historical-mode rolling fit ({note}): "
        f"{len(flagged)} of {len(rows)} break-even parameters flagged unidentified"
        + (f" ({', '.join(flagged)})." if flagged else ".")
    )


def _ssr_sentence(results: Results) -> str:
    rows = results.rows("ssr")
    parts = []
    for r in rows:
        v, se = _v(results, "ssr", r, "realised_last")
        fo, _ = _v(results, "ssr", r, "first_order_median")
        tgt, _ = _v(results, "ssr", r, "target")
        if math.isfinite(v):
            parts.append(
                f"at {r} the last realised SSR is {_fmt(v, se, 2)} against the target {tgt:g} "
                f"and the fit's first-order {_fmt(fo, digits=2)}"
            )
        else:
            parts.append(f"at {r}: {_note(results, 'ssr', r, 'realised_last')}")
    return "Realised SSR: " + "; ".join(parts) + "."


def _window_sentence(results: Results) -> str:
    if "window" not in results.tables() or "first_close" not in results.columns("window"):
        return "Fewer than two marked closes: no window statistics."
    row = results.rows("window")[0]
    n, _ = _v(results, "window", row, "n_closes")
    first, _ = _v(results, "window", row, "first_close")
    last, _ = _v(results, "window", row, "last_close")
    ret, _ = _v(results, "window", row, "return_pct")
    vol, _ = _v(results, "window", row, "realised_vol_pct")
    return (
        f"Over the {int(n)} marked closes ({_note(results, 'window', row, 'n_closes')}) {row} "
        f"went from {first:.2f} to {last:.2f} ({ret:+.2f}%), with a close-to-close realised "
        f"volatility of {vol:.1f}% (252-day annualisation)."
    )


def _missing_sentence(results: Results) -> str:
    if "pnl_missing" not in results.tables():
        return ""
    rows = [r for r in results.rows("pnl_missing") if r != "none"]
    if not rows:
        return "Every stored date carries its P&L."
    lines = [f"{len(rows)} date(s) lack some or all of their P&L (never dropped silently):", ""]
    for r in rows:
        n, _ = _v(results, "pnl_missing", r, "rows_without_pnl")
        failed, _ = _v(results, "pnl_missing", r, "failed")
        why = _note(results, "pnl_missing", r, "rows_without_pnl")
        if failed == 1.0:
            lines.append(
                f"- {r}: failed, no rows stored ({int(n)} live trade(s) without a value or P&L); "
                f"{why}"
            )
        else:
            lines.append(f"- {r}: {int(n)} row(s) without P&L; {why}")
    return "\n".join(lines)


def _book_sentence(results: Results) -> str:
    """The cumulative desk P&L of the fixed trades, one sentence per unit (P2: a book adds only
    trades quoted in one unit)."""
    if "pnl_trade" not in results.tables() or "pnl" not in results.columns("pnl_trade"):
        return "No stored date carries a P&L yet."
    units = _units(results, "pnl_trade", "pnl")
    out = []
    for unit, rows in units.items():
        books = [r for r in rows if r.startswith("book (")]
        fixed = [r for r in rows if r.startswith("fixed:")]
        subject = books[0] if books else (fixed[0] if len(fixed) == 1 else "")
        if not subject:
            continue
        v, se = _v(results, "pnl_trade", subject, "pnl")
        ex, ex_se = _v(results, "pnl_trade", subject, "explained")
        rc, rc_se = _v(results, "pnl_trade", subject, "recalibration")
        rs, rs_se = _v(results, "pnl_trade", subject, "residual")
        members = _note(results, "pnl_trade", subject, "pnl").split(";")[0]
        members = members.replace("fixed:", "")
        who = (
            f"the fixed trades quoted in {unit} ({members})"
            if books
            else f"`{subject}` (the only fixed trade quoted in {unit})"
        )
        out.append(
            f"- {who}: cumulative desk P&L {_fmt(v, se)} x100 {unit}, of which the Greeks explain "
            f"{_fmt(ex, ex_se)}; recalibration {_fmt(rc, rc_se)}, residual {_fmt(rs, rs_se)}."
        )
    if not out:
        return "No fixed trade carries a P&L yet."
    return "\n".join(["The fixed book, unit by unit (no total adds two units):", "", *out])


def _stderr_sentence(results: Results) -> str:
    """What the cumulative standard errors are (from :func:`aggregate`'s notes)."""
    if "pnl_trade" not in results.tables() or "pnl" not in results.columns("pnl_trade"):
        return ""
    long = results.long("pnl_trade", "pnl")
    paired = [str(r) for r, n in zip(long["row"], long["note"]) if SE_PAIRED_CUMULATIVE in str(n)]
    rss = [str(r) for r, n in zip(long["row"], long["note"]) if SE_RSS_DATES in str(n)]
    head = (
        f"The standard error of the cumulative P&L of {len(paired)} trade(s) is the direct "
        "paired one of V(last) - V(inception) + cash flows (each date stores it for its "
        "attributed trades, at one extra pricing per trade and date). "
        if paired
        else "This store predates the stored cumulative P&L error. "
    )
    tail = (
        f"For the other {len(rss)} P&L row(s) of the table, the books, the buckets, the months "
        "and the daily book, the error shown is the root sum of squares of correlated daily "
        "errors - not the error of the cumulative P&L: the dates share one seed, and so do the "
        "trades of one date, and no pricing-free estimator gives their covariance."
        if rss or not paired
        else "The books, the buckets, the months and the daily book show the root sum of squares "
        "of correlated errors - not the error of their sum."
    )
    return head + tail


def _legacy_sentence(results: Results) -> str:
    """The dates whose stored record cannot confirm their leverage's numbers (never silent)."""
    n, _ = _v(results, "setup", "backtest", "n_legacy_unverified")
    if not math.isfinite(n) or n == 0:
        return ""
    dates = _note(results, "setup", "backtest", "n_legacy_unverified")
    return (
        f"**{int(n)} date(s) carry a {LEGACY_UNVERIFIED}**: {dates}. They were written before "
        "the record held the digest of the leverage's numbers, so their verdict checks the "
        "leverage by key and completeness only (a cache entry rewritten with other numbers "
        "under the same key would go unnoticed for these dates; a date recorded at version 2 "
        f"that uses their state does check those numbers). Note: {LEGACY_RECOMPUTE}; "
        "`volsto-backtest status` prints the command. The manifest lists them "
        "(`backtest_legacy_unverified`)."
    )


def _repair_sentence(results: Results) -> str:
    """Whether the marked window exercises the calendar-repaired surfaces."""
    n, _ = _v(results, "fits", "calendar repaired", "n_dates")
    if not math.isfinite(n):
        return ""
    if n == 0:
        return (
            "No marked date of this window is calendar-repaired, so the backtest does not "
            "exercise the repaired-surface path of stage 1 here (in 2022 H2 the 30 repaired days "
            "run from 2022-09-09 to 2022-12-30, SPEC §13.1 after the θ fix of 2026-09-17; a shard "
            "over 2022-09-09..2022-09-13 would)."
        )
    return f"{int(n)} marked date(s) use a calendar-repaired surface."


def narrative(results: Results) -> str:
    n_dates, _ = _v(results, "setup", "backtest", "n_dates")
    n_ok, _ = _v(results, "setup", "backtest", "n_ok")
    fast, _ = _v(results, "setup", "backtest", "fast")
    rng = _note(results, "setup", "backtest", "n_dates")
    name = _note(results, "setup", "backtest", "fast")
    parts = [
        "## Scope",
        "",
        _legacy_sentence(results),
        "",
        f"Backtest `{name}` over {int(n_dates)} vendor dates ({rng}) of the "
        f"{_note(results, 'setup', 'backtest', 'data')}; {int(n_ok)} are marked"
        + (
            f" ({_note(results, 'setup', 'backtest', 'n_ok')})."
            if _note(results, "setup", "backtest", "n_ok")
            else "."
        ),
        "",
        _window_sentence(results),
        "",
        "A single window this short is a proof of concept, too short for a conclusion about the "
        "models. The multi-year run waits on the paid end-of-day archive; it needs only a config "
        "change - `dates.start`, `dates.end` and `data.root` - plus a new store (`paths.out` "
        "and `paths.snapshots`, or `--out` and `--snapshots`): `dates.start` is part of the "
        "config hash, so `run` refuses the old store (`dates.end` and `data.root` are not "
        "hashed; the inputs digest of each date follows its day files).",
        "",
        "{{table:window}}",
        "",
    ]
    skipped_note = _note(results, "setup", "backtest", "n_skipped")
    if skipped_note:
        parts += [
            f"Dates dropped by `data.missing_close: skip_date`: {skipped_note}. The trading-day "
            "fixing grid of the term sheets (k/252 years is the k-th calendar date after "
            "inception) runs over the remaining vendor dates, so every later fixing of a trade "
            "spanning a gap lands one exchange day late and the return across the gap counts as "
            "one daily return; the rows of those trades list their gaps.",
            "",
        ]
    if fast >= 1:
        parts += [
            "Fast mode: toy particle and path counts on a handful of dates and short-maturity "
            "term sheets. The numbers check the plumbing, not the models.",
            "",
        ]
    npart, _ = _v(results, "setup", "backtest", "n_particles")
    npaths, _ = _v(results, "setup", "backtest", "n_paths")
    tgt, _ = _v(results, "setup", "backtest", "ssr_target")
    eps, _ = _v(results, "setup", "backtest", "skew_eps")
    parts += [
        "## Configuration",
        "",
        f"- marking fit at SSR target {tgt:g} and skew eps {eps:g}, stage 3 off (the fit's "
        "first-order SSR is what is marked);",
        f"- leverage calibrated per date at {int(npart)} particles "
        f"({_note(results, 'setup', 'backtest', 'n_particles')}), cached by key;",
        f"- pricing on {int(npaths)} paths, {_note(results, 'setup', 'backtest', 'n_paths')};",
        f"- attribution: {_note(results, 'setup', 'backtest', 'pillars')}. The sticky-leverage "
        "Greeks are model Greeks with the previous date's leverage frozen in forward "
        "log-moneyness, not recalibrated Greeks; the leverage refit shows only in the "
        "recalibration bucket.",
        f"- variance products (variance swap, KO variance swap): "
        f"{_note(results, 'setup', 'backtest', 'variance_trades')}.",
        "- rolling book: "
        + (
            "marked on every date after inception (paired P&L)."
            if _v(results, "setup", "backtest", "rolling_daily")[0] >= 1
            else "priced on its inception dates only (the inception pricing; the fixed book "
            "carries the mark-to-market drift)."
        ),
        "",
        "## Book and inception prices",
        "",
        "Every value is x100 in its trade's own unit: % of notional (notes, cliquet), % of the "
        "inception spot (vol knock-out put), vol points of vega notional 1 (variance swaps); "
        "the tables are split by unit and no total adds two units.",
        "",
        "{{tables:trades}}",
        "",
        "{{tables:inception}}",
        "",
        "## P&L attribution",
        "",
        f"Sign convention: {DESK_CONVENTION}. Every P&L number below is a desk P&L.",
        "",
        _book_sentence(results),
        "",
        "Per trade the buckets, the recalibration, the residual and the cash flows sum exactly "
        "to the P&L of each day. The rates bucket is the first-order P&L along each day's actual "
        "rate and dividend curve moves (a directional sensitivity), not a parallel shift.",
        "",
        _stderr_sentence(results),
        "",
        _missing_sentence(results),
        "",
        "{{table:pnl_missing}}",
        "",
        "{{tables:pnl_trade}}",
        "",
        "{{table:pnl_bucket}}",
        "",
        "{{tables:pnl_month}}",
        "",
        "{{figure:book_pnl}}",
        "",
        "{{figure:trade_pnl}}",
        "",
        "{{figure:explained}}",
        "",
        "## Fitted parameters and stability",
        "",
        "The marking fit carries no standard errors (its targets are deterministic functions of "
        "the day's surface), so the stability flags on the marking parameters report the daily "
        "changes without a band. The flags of SPEC 15 Part 4 come from the historical-mode "
        "rolling fit on the stored pillar history, configured by `stability.fit`: "
        f"{_note(results, 'setup', 'backtest', 'stability_fit')}.",
        "",
        "{{table:params_summary}}",
        "",
        "{{table:params_flags}}",
        "",
        _stability_sentence(results),
        "",
        "{{table:stability_hist}}",
        "",
        "{{figure:params}}",
        "",
        "## Realised against marked SSR",
        "",
        f"The realised SSR needs {int(_v(results, 'setup', 'backtest', 'ssr_window')[0]) + 1} "
        "dates before its first estimate (earlier dates are missing, with the reason, in the "
        "results).",
        "",
        _ssr_sentence(results),
        "",
        "{{table:ssr}}",
        "",
        "{{figure:ssr}}",
        "",
        "## Vol knock-out put",
        "",
    ]
    vko_rows = results.rows("vko") if "vko" in results.tables() else []
    for r in vko_rows:
        if r == "none":
            continue
        outcome = _note(results, "vko", r, "inception_value")
        rv, _ = _v(results, "vko", r, "realised_vol")
        ko, _ = _v(results, "vko", r, "vol_ko")
        realised = (
            f"realised vol to date {rv:.2f} vol pts"
            if math.isfinite(rv)
            else "no realised return yet (inception date only)"
        )
        parts.append(
            f"- `{r}`: {realised} against the {ko:.1f} vol pts barrier; realised outcome {outcome}."
        )
    parts += [
        "",
        "{{table:vko}}",
        "",
        "{{figure:vko}}",
        "",
        "## Fits and cost",
        "",
        "{{table:fits}}",
        "",
        _repair_sentence(results),
        "",
        "{{table:cost}}",
    ]
    # place only the tables and figures this results set declares (a window without any
    # P&L, or fewer than two marked dates, has no book, monthly or window tables); a
    # ``{{tables:<name>}}`` line stands for every table of that family (one per unit)
    names = [t.name for t in tables(results)]
    declared = {f"{{{{table:{n}}}}}" for n in names}
    declared |= {f"{{{{figure:{f.name}}}}}" for f in figures(results)}
    kept: list[str] = []
    for x in parts:
        family = _FAMILY.fullmatch(x)
        if family is not None:
            base = family.group(1)
            members = [n for n in names if n == base or n.startswith(base + "_")]
            kept += [line for n in members for line in (f"{{{{table:{n}}}}}", "")][:-1]
            continue
        if not (_PLACEHOLDER.fullmatch(x) and x not in declared):
            kept.append(x)
    return "\n".join(kept)


_PLACEHOLDER = re.compile(r"\{\{(table|figure):[A-Za-z0-9_]+\}\}")
_FAMILY = re.compile(r"\{\{tables:([A-Za-z0-9_]+)\}\}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
