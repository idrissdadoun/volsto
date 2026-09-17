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
