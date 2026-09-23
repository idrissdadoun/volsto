# Contributing to volsto

The standing rules the project is built under. `SPEC.md` is the design; this file is how the
work is done. Every rule here was set by the owner and each has a reason recorded beside it.

## Evidence

- **Every Monte Carlo number carries its standard error.** In code (`<name>_stderr`), in a
  table (value ± se), in a figure (an error bar, a band, or a stated reason why none), in a
  commit message and in a report. A number without its error is not reported.
- **Every report states its wall clock and whether it recalibrated.**
- **If a formula or a test fails, report the discrepancy. Do not patch the test to pass.** A
  measurement that contradicts the hypothesis it was sent to test is a result: say so plainly,
  with the numbers.
- **Never assert on wall clock.** Print it; do not test it.
- **No silent defaults.** Every default is a named, documented constant or a YAML value with its
  reason.

## Calibration and the cache

- **Tests read the leverage cache and never calibrate** (`allow_calibrate=False`; a cache miss
  skips with the reason). The sanctioned exceptions are the session fixtures that build toy
  stores into temporary directories, once per pytest run, at 2·10⁴ particles: `toy_build` and
  `toy_marking_build` in `tests/conftest.py`, and `toy_backtest_build` in
  `tests/_backtest_build.py` (imported by `tests/test_backtest.py`; it runs `volsto-backtest run`,
  the one M10 path that calibrates).
- **A toy build that raises, exits non-zero or whose builder dies fails** every test that requires it
  (`ToyBuild.require()`); it never skips (the build has no optional input) and its wall clock is
  printed, never used to skip. Only an absent optional input (the HDN sample, the Tectonic engine)
  may skip, with its reason.
- **Scripted and agent `volsto-backtest` runs set `VOLSTO_BACKTEST_REQUIRE_PATHS=1`**, so a command
  without explicit `--out`/`--cache`/`--snapshots` is refused instead of writing into the
  repository's `outputs/` and `cache/` (an M10 verifier once did).
- **Studies never calibrate.** `volsto-study run` and `render` forbid calibration at the entry of
  `calibrate_leverage` (`volsto/calibration/guard.py`, inherited by child processes) and record
  whether one started; `volsto-backtest run` is the only M10 command that calibrates.
- **Viewers and the read API never calibrate and never simulate.** A missing point prints the
  exact `volsto-precompute` command that produces it. `volsto-precompute` is the only place in
  the viewers layer that calibrates, and it says so in its log and manifest.
- **Production particle count is 8·10⁵** (SPEC §11).

## Code

- `black` (line length 100), `ruff`, `mypy --strict` clean on every file touched.
- `pytest -n auto -m "not slow"` for the fast suite; the slow tests run serially with `-s`.
- Long runs under `caffeinate -i`, and one Monte Carlo study process at a time on a laptop: a
  recalibration run can hold 9–13 GB.
- The book PDF under `docs/` is copyrighted and git-ignored. Never commit it.
- Commit at the end of every green milestone.

## When the same class of defect appears a third time

**Stop fixing instances. Find the invariant that makes the class impossible, enforce it in one
place, and add the test that walks every site against it.** (Owner's rule, 2026-09-16.)

Fixing the sites a reviewer lists leaves the sites nobody listed. The rule exists because of the
M9 viewers: six independent reviews and four fix-and-verify rounds found the same defect fifteen
times — a Monte Carlo number drawn with a NaN, zero or absent error bar, or a page crashing on a
table written without its stderr twin — because each round fixed the instances it was handed.
What closed it was structural:

1. **One implementation of the invariant.** `mc_rows` / `missing_mc_notice` /
   `nothing_plottable` in `volsto/viewers/pages/_common.py` decide what a figure may draw; the
   derived helpers return `(frame, dropped, absent)` so a caller cannot plot what it cannot
   annotate.
2. **Every site declares itself.** Each page lists its `(value, stderr)` pairs in `MC_PAIRS`.
3. **A test walks every declared site against the real data** — the read API's actual output
   on the repository's store, cache and outputs — so the next rename or new column fails a test
   instead of a page.
4. **An adversarial sweep confirms the class is gone**, not just the listed instances: 248
   headless renders over pathological inputs (the twin dropped, NaN on one row and on every row,
   the value NaN, the column dropped, the table empty) found zero survivors.

How to apply it: count occurrences across review rounds, not within one. On the third, before
writing another fix, write down the invariant in one sentence ("no page draws a Monte Carlo
number without a finite stderr"), find the single place that can enforce it, and write the walking
test first — it will fail on every unlisted site, which is the point.

Applied since (M10, 2026-09-16/17):
- **Importer tag (2026-09-22):** a stored artefact must not outlive the code that produced it —
  the calibration code tag (M4), the golden legacy backtest store (2026-09-19) and the backtest's
  snapshot binding (2026-09-22: 25 proof-of-concept dates read `done` under a changed importer)
  were three occurrences. One place: `IMPORTER_TAG` in `volsto/market/import_hdn.py`, written into
  every snapshot's provenance, checked by the store's verdict and by `snapshot_bound`, and hashed
  against the importer, surface and curve sources (`importer_guard.json`,
  `test_importer_tag_guard`). Bump it when a change moves any snapshot; re-record the hash without
  a bump only for a change proven not to, and say so in the commit.
- **Studies:** every catalogue study declares its exact (error-free) numbers by kind in
  `EXACT_KINDS`; the walker in `volsto/studies/catalogue/_common.py` (run by
  `tests/test_catalogue_s1_s4.py` over every fast config) refuses any undeclared exact number, and
  `results.py` refuses a Monte Carlo number without a finite stderr.
- **Aggregate errors:** the backtest's stage 2 computes every aggregate error in one helper
  (`aggregate()`), which uses a paired error where one is stored and labels every root sum of
  squares of correlated errors as such.
- **Backtest storage:** after four rounds of "results destroyed / a done date that does not follow
  from its inputs", the store became immutable attempts plus one atomic pointer per date, read by
  one verdict function, with a walking test over every recorded input and a crash matrix over every
  file operation.
