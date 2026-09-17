# The study catalogue

A study is a YAML config plus a runner module. `volsto-study run <config>` turns it into a
self-contained output directory whose tables and figures regenerate from its
`results.parquet`. This page gives, for each catalogue study (S1–S7) and for the rolling
backtest:

- the question it answers;
- what it reads;
- how to run it;
- whether it can run on today's repository state.

The design record is SPEC §10.1 (runner), §10.2 (catalogue) and §10.3 (backtest). The model and
the conventions (units, signs, rota, SSR) are in [methodology.md](methodology.md). Run every
command from the repository root with the virtual environment active.

"Today" below means the repository state of 2026-09-16. It was checked by evaluating each full
config's requirements against the repository's store, cache and outputs, without computing
anything.

---

## How a study runs

```bash
volsto-study list                                   # every config, its mode and its question
volsto-study run configs/studies/catalogue/<id>.yaml [--fast] [--out DIR]
    [--grid G] [--store S] [--cache C] [--outputs O] [--set key.sub=value ...]
    [--profile] [--no-latex-check]
volsto-study render <output_dir>                    # rebuild tables, figures, study.md/.tex/.pdf
volsto-study rerun  <output_dir> [--nse 2]          # re-execute the stored config and diff
```

**Inputs.** A config names four roots:

| Key | Default | Contents |
|---|---|---|
| `grid` | varies | a `volsto-precompute` grid, or `null` |
| `store` | `outputs/store` | the results store written by `volsto-precompute` |
| `cache` | `cache` | the leverage cache |
| `outputs` | `outputs` | the M7 / M8b / Part 0 artefacts, read only |

A study reads three kinds of requirement:

- store **points** (grid points computed by `volsto-precompute`);
- cached **leverages**;
- **artefacts** under the outputs root.

Paths in a YAML resolve against the repository root. The `--grid/--store/--cache/--outputs/--out`
flags resolve against the current directory.

**No calibration.** A study never calibrates. `calibrate_leverage` refuses while `run` or
`render` holds the guard, in every thread and child process, and a refusal fails the run
(SPEC §10.1). `manifest.json` records `recalibrated: false`, and whether another process changed
the cache during the run.

**Exit status.**

| Code | Meaning |
|---|---|
| 0 | The study ran. For `rerun`: nothing moved. |
| 1 | An error: a usage error, an exception, a failed LaTeX check, or a calibration attempt. For `rerun`: a number moved by more than 2 standard errors, or was added, removed or changed unit. |
| 2 | A requirement is missing. The run prints every missing item and the exact command that produces it, then stops before computing anything. No output directory is created. |

Point and leverage ids of the configured grid are printed as one runnable line of the form
`volsto-precompute --grid G --store S --cache C --only <ids> --resume`. An artefact prints its
own command.

**Full and fast.** Each study has a full config `<id>.yaml` and a CI-fast `<id>_fast.yaml`
(`mode: fast`, small budgets, toy inputs). A fast run's `study.md` and `study.tex` carry the
banner "FAST MODE — reduced paths and toy inputs: plumbing check, not results." `--fast` turns
any config into a fast run.

**Output directory.** The default is `outputs/studies/<name>`; `--out` moves it. A run is built
in a staging directory next to the target and moved into place, so a failed run leaves nothing
behind.

```
<out>/
  results.parquet      every number: value, stderr, unit, source (store:/cache:/artefact:/computed)
  tables/<name>.tex    booktabs; tables over 25 rows split into <name>-partK.tex
  figures/<name>.pdf   and <name>.png (house style)
  study.md             title, the question, the narrative with tables inlined, provenance
  study.tex            master LaTeX document
  study.pdf, study.log compiled by Tectonic (the LaTeX check; skipped with --no-latex-check,
                       or when Tectonic is not installed — `brew install tectonic`)
  manifest.json        config and its hash, git commit and dirty flag, code tags, grid, cache keys,
                       store points, artefacts with sha256, particles, seeds, paths, wall clock
                       per phase, recalibrated, host and library versions
  rerun/<utc>/         written by `volsto-study rerun`: the rerun's files plus diff.csv, diff.md
```

- **`render`** rebuilds `tables/`, `figures/`, `study.md` and `study.tex` byte-identically from
  `results.parquet` and the stored manifest.
- **`rerun`** executes `manifest["config"]`, never the YAML on disk. A Monte Carlo number has
  moved when $|\Delta| > 2\max(\text{se}_{\text{old}}, \text{se}_{\text{new}})$. The two runs share
  their seeds, so the errors are not combined in quadrature. An exact number has moved when it
  changes by more than $10^{-12}$ relative (SPEC §10.1).

**Errors in the catalogue** (SPEC §10.2). Every Monte Carlo number carries its standard error.
Models priced inside a study share one step schedule and one seed, so an LSV − LV difference
priced there carries a **paired** error; the quadrature error is shown beside it. A difference of
two stored estimates (S1, and S4 when read from the store) carries the quadrature error, labelled
"not the exact error": the estimates share random numbers, and quadrature bounds the exact error
only when their correlation is non-negative, which no study checks. The forward 90/110 skew of
stored smiles (S4, S5) carries the sum of the two strikes' errors, a bound whatever the
correlation.

---

## Status today

| Study | Full config | Today | What unblocks it |
|---|---|---|---|
| S1 | `s1.yaml` | runs (5 store points) | — |
| S1 | `s1_grid.yaml` | exits 2: 87 leverages missing | the default grid's 1F sweep (VM run, [vm_grid_run.md](vm_grid_run.md)) |
| S2 | `s2.yaml` | runs (4 leverages + the repaired history) | — |
| S3 | `s3_headline.yaml` | runs (4 leverages) | — |
| S3 | `s3.yaml` | exits 2: 31 of 35 leverages missing | precompute `configs/grids/s3_ko_var.yaml` |
| S4 | `s4.yaml` | runs (5 store points); the LSV forward-skew exposure is reported missing | `volsto-precompute ... --risk light` on the LSV points |
| S5 | `s5.yaml` | exits 2: the 36 marking points are missing | the default grid's marking points |
| S6 | `s6.yaml` | runs (12 artefacts) | — (its claim follows study C's table) |
| S7 | `s7.yaml` | runs (4 artefacts) | — |
| Backtest | `backtest_2022h2.yaml` | exits 2: 102 of 127 dates missing (25 computed, 2022-07-01..2022-08-05) | `volsto-backtest run configs/backtest/hdn_2022h2.yaml --resume` |
| Backtest PoC | `backtest_2022h2_poc.yaml` | runs (25 dates; 2453 numbers, 22 tables, 6 figures) | — (stage 1 of the window is done) |

**Fast configs.**

- S1–S4 fast ran on the README quickstart's toy store (`configs/grids/toy.yaml`) on
  2026-09-16. Runner wall clocks: S1 2.4 s, S2 6.1 s, S3 7.9 s, S4 13.2 s.
- S6 and S7 fast ran on the repository's artefacts (`--outputs outputs`) in 2.0 s and 1.7 s.
- S5 fast and the backtest fast config each need their own toy build, which calibrates. See their
  sections below; those two builds were not run for this page.

---

## S1 — forward vol and cliquets vs vol-of-vol

**Question.** How do the forward smile, the forward volatilities and the capped cliquets move
with vol-of-vol when every model is calibrated to the same smile?

**What it does** (runner `volsto.studies.catalogue.s1_forward_vol`; SPEC §10.2). S1 regenerates
the original paper's tables on the placeholder surface. It reads from the store:

- the M4 headline cells;
- the forward smiles and forward vols of every model: LV, 1F $\nu$ grid, 2F presets.

A point missing from the store but with a cached leverage is priced with the precompute's own
functions (`price_missing`; `--set price_missing=false` gives a store-only run). The study
reports:

- the headline table;
- forward vols per window;
- the per-strike LSV − LV forward smile, with error bounds (put-wing invariance);
- the study cliquet's ladder at its fixed 2 % cap across models;
- the regeneration of `tests/test_m4_regression.py`, read as data.

The original study's archive is absent, so the placeholder baseline stands in for it, and
`study.md` says so.

**Run.**

```bash
volsto-study run configs/studies/catalogue/s1.yaml        # LV, 1F ω = 1/2/3, 2F Table 8.2
volsto-study run configs/studies/catalogue/s1.yaml --set price_missing=false   # store only
volsto-study run configs/studies/catalogue/s1_grid.yaml             # the whole default-grid sweep
volsto-study run configs/studies/catalogue/s1_fast.yaml --grid configs/grids/toy.yaml \
    --store <toy store> --cache <toy cache> --outputs <dir> --out <dir>
```

**Status.**

- `s1.yaml` runs today on `outputs/store` (grid `configs/grids/placeholder_cached.yaml`). The
  recorded regeneration found 80/80 baseline keys within the regression tolerance, with max
  $|d|/\text{se}$ = 0.0075, in 2.5 s with nothing priced. The store and the baseline are the same
  computation (SPEC §10.2).
- `s1_grid.yaml` needs the default grid's 90 one-factor LSV points. Three are cached (the
  $\omega$ = 1, 2, 3 headline points), so it exits 2 and prints the
  `volsto-precompute --grid configs/grids/default.yaml ... --only <87 ids> --resume` line until
  the VM grid run lands.

---

## S2 — the volatility knock-out put

**Question.** How much cheaper than the vanilla is the vol knock-out put across barriers and
models, and where does each barrier sit in the in-the-money realised-vol distribution that sets
the sign of the LSV-minus-LV difference?

**What it does** (runner `volsto.studies.catalogue.s2_vko`; SPEC §10.2, §6.2). S2 prices the 12m
100 % VKO put for every model from its cached leverage, because the store does not hold the
in-the-money distribution or its errors. It reports:

- the VKO/vanilla ratio over `vol_ko` ∈ {20, 25, 30, 35, 40} %;
- prices and P(KO);
- the full-life realised-vol percentiles conditional on $S_T < K$, with a paired pair bootstrap;
- $P(\sigma_{\text{real}} > \text{vol\_ko} \mid \text{ITM})$ per barrier;
- a store cross-check.

The 2022 H2 case reads the repaired surface history (artefact
`essvi_gate/hdn_history_repaired.csv`). It gives the realised vol to date, the budget used, the
knock-out state, the largest vol that keeps the put alive, and the start-date marks. The 12m
payoff itself is not observable inside the window.

**Run.**

```bash
volsto-study run configs/studies/catalogue/s2.yaml        # 4·10⁵ paths; ≈ 113 s (SPEC §10.2)
volsto-study run configs/studies/catalogue/s2_fast.yaml --grid configs/grids/toy.yaml \
    --store <toy store> --cache <toy cache> --outputs outputs --out <dir>
```

If the history artefact is missing, the run exits 2 and prints
`.venv/bin/python scripts/m7_hdn_history.py --tag _repaired --out outputs/essvi_gate`.

**Status.** `s2.yaml` runs today.

---

## S3 — conditional and knock-out variance

**Question.** Which conditional and knock-out variance strikes does vol-of-vol move at a fixed
smile, and which does Gyongy's theorem pin to the local-vol value?

**What it does** (runner `volsto.studies.catalogue.s3_conditional_variance`; SPEC §10.2, §6.1).
On one path set per model, S3 prices:

- the variance swap;
- up, down and corridor variance at B ∈ {90, 100, 110} %;
- the convexity spread;
- the KO variance swap at {105, 110, 120} %: strike, P(KO), $E[\tau/N]$ and the paired
  $K_{KO} - K_{\text{var}}$.

The Gyöngy check is computed per cell against a tolerance equal to the model's own variance-swap
residual (a heuristic, not a bound). With daily fixings the previous-close legs are pinned only
up to $O(\Delta t)$ plus the calibration residual. The KO LSV − LV strike is swept against
$\nu$, $\rho$ and $\theta$. `study.md` records that $K_{KO} > K_{\text{var}}$ holds under
negative skew and a non-inverted term structure but is not a theorem.

**Run.**

```bash
# the five headline models; runs today (162 s at 4·10⁵ paths, SPEC §10.2)
volsto-study run configs/studies/catalogue/s3_headline.yaml
volsto-study run configs/studies/catalogue/s3.yaml            # + the ρ and θ sweeps
volsto-study run configs/studies/catalogue/s3_fast.yaml --grid configs/grids/toy.yaml \
    --store <toy store> --cache <toy cache> --outputs <dir> --out <dir>
```

**Status.**

- `s3_headline.yaml` runs today. It found 0 of 36 cells beyond the model's floor, and 25
  significant but within it (SPEC §10.2).
- `s3.yaml` reads `configs/grids/s3_ko_var.yaml`: the default grid's 1F specs at $\kappa$ 1.5,
  plus Table 8.2 at $\theta$ ∈ {0.1, 0.245, 0.4, 0.6, 0.8}. 31 of its 35 leverages are missing, so
  it exits 2 with `volsto-precompute --grid configs/grids/s3_ko_var.yaml ... --only <31 ids>
  --resume`. That precompute projects to 4.46 h at 2 threads and 4.04 h at 12
  (`volsto-precompute --grid configs/grids/s3_ko_var.yaml --dry-run`, SPEC §10.2).

---

## S4 — autocall and Phoenix model risk

**Question.** How much do the 3y autocall and Phoenix prices, knock-in probabilities and
expected lives move from local vol to stochastic-local vol at a fixed smile, which legs carry
the difference, and how exposed is each leg to forward skew?

**What it does** (runner `volsto.studies.catalogue.s4_autocall`; SPEC §10.2, §6.6). S4 reads the
store's M6 cells, which are stored as fractions under a "% notional" label; it converts them and
says so. A point without cells is priced on the shared schedule. The study reports:

- price, expected life, P(KI), P(breach) and the autocall probabilities;
- the paired LSV − LV difference by leg;
- the forward 90/110 skew per window over the spot 1y skew;
- a `horizon` table disclosing every number that lies beyond a leverage's calibration horizon.

The forward-skew exposure per leg is computed under local vol. For an LSV it would need a
leverage recalibrated on every bumped surface, so the LSV `skew_T` is read from the store's risk
tier.

**Run.**

```bash
volsto-study run configs/studies/catalogue/s4.yaml        # full run ≈ 4 min (SPEC §10.2)
volsto-study run configs/studies/catalogue/s4_fast.yaml --grid configs/grids/toy.yaml \
    --store <toy store> --cache <toy cache> --outputs <dir> --out <dir>
```

**Status.** `s4.yaml` runs today. The LSV `skew_T` rows stay missing until the store is
refreshed with the line the study prints:
`volsto-precompute --grid configs/grids/placeholder_cached.yaml --store outputs/store --cache
cache --only <LSV point ids> --resume --risk light`. SPEC §10.2 puts that at about 0.8–1.6 h per
LSV point. In fast mode the 3y notes run on the toy leverage's last slice, held
beyond its 1y horizon: plumbing, not numbers.

---

## S5 — the marking dials

**Question.** What does the choice of SSR mark (ssr_target, skew_eps) cost or earn per product
on each snapshot, and what does each mark imply for the leverage, the realised SSR and the
forward skew?

**What it does** (runner `volsto.studies.catalogue.s5_marking`; SPEC §10.2, §15 Part 3).

- **Binding map, fitted inline.** `fit_2f_marking` without stage 3, over `ssr_target` ∈
  {0.75, …, 2.0} × `skew_eps` ∈ {0.05, 0.10, 0.20, 0.30} on each SPX snapshot. This is a
  parameter fit of seconds, not a leverage calibration. It reports status, parameters, the gap and
  binding edge at the constraint maturities, the $\nu$-box, $\nu$-cap and $\rho_{12}$ flags, and
  the largest first-order SpotVolCovar miss.
- **From the store's marking points** (`marking:<surface>:ssr<s>:eps<e>`):
  - mean $|L-1|$;
  - the realised LSV SSR ± se beside the first-order SSR;
  - the forward 90/110 skew against the spot skew;
  - product prices;
  - the light-tier Greeks of the autocall and cliquet.
- **Cost of a mark.** Its price minus the price at the reference mark (1.0, 0.10), with
  errors in quadrature.

**Surface.** The SPX 2022-12-30 surface is the committed plain-SSVI snapshot
`configs/surfaces/snapshots/hdn_2022H2_ssvi/spx_2022-12-30.yaml` (a single $\rho$,
`essvi: false`), not the repaired eSSVI surface of M10 Part 0
(`outputs/essvi_gate/snapshots/spx_2022-12-30.yaml`; that day needed the calendar repair). The
default grid's `spx_2022-12-30` entry points at it, and so does the SPX snapshot of the M7 greek
and of M8b (`volsto.studies.m8b.SPX_SNAPSHOT`, `scripts/m7_p1_marking.py`). On that day the
2y / 3y ATM skew is −0.2181 / −0.1910 under SSVI against −0.1940 / −0.1660 repaired
(SPEC §13.1). The other two grid snapshots, `spx_2022-09-15` and `spx_2022-12-02`, are eSSVI
and were not changed by the repair. Whether to switch to the repaired surface is the owner's
decision; a switch changes the marking and LV cache keys and needs M8b studies C and D and
S5–S7 re-run.

**Run.**

```bash
volsto-study run configs/studies/catalogue/s5.yaml
```

The fast config runs on the toy marking grid: four marking fits on the placeholder surface at
$2\cdot10^4$ particles, which **calibrates**. The test suite builds it in
`tests/conftest.py::toy_marking_build`. By hand, with a separate store:

```bash
volsto-precompute --grid configs/grids/toy_marking.yaml --store <marking store> --cache <toy cache>
volsto-study run configs/studies/catalogue/s5_fast.yaml --grid configs/grids/toy_marking.yaml \
    --store <marking store> --cache <toy cache> --outputs outputs --out <dir>
```

That two-line recipe was not run for this page. The placeholder surface is unsuitable for
marking work: every toy fit binds at the $\nu$ cap (methodology, section 9, item 4).

**Status.** `s5.yaml` exits 2 today, because the store holds none of the default grid's 36
marking points (3 SPX snapshots × 4 × 3 marks). It prints
`volsto-precompute --grid configs/grids/default.yaml --store outputs/store --cache cache --only
marking:spx_2022-12-30:ssr0.75:eps0.05 … --resume` (36 ids). The fast run takes about 17 s once
its store exists (SPEC §10.2).

---

## S6 — shadow rotation

**Question.** Does the static M7 shadow-rotation greek predict the simulated M8b recalibration
P&L of each re-marking policy in the desk convention, and how nonlinear is that P&L at +2 and +3
rota?

**What it does** (runner `volsto.studies.catalogue.s6_shadow_rotation`; SPEC §10.2, §8.2, §15
Part 3). S6 reads artefacts only:

- the M7 greek (`m7/p1_marking_shadow_rotation.csv`);
- the static greek files (`m8b/C/static_*.json`);
- study C's table.

It recomputes the ratio ± se, z and the nonlinearity, and carries the refit flags and the table's
own state: modification time, missing rows, and task results newer than the table. Each ratio is
classed at 2 se against $1 \pm 0.30$ as above, below, within or undecided. A policy column enters
the claim only when it is clean. The claim is stated per rota as a desk loss or gain.

**Surface.** The M7 greek and study C were computed on the SPX 2022-12-30 plain-SSVI snapshot
(see S5, "Surface").

**Run.**

```bash
volsto-study run configs/studies/catalogue/s6.yaml
volsto-study run configs/studies/catalogue/s6_fast.yaml --outputs <outputs root> --out <dir>
```

The fast config is written for the synthetic outputs of `tests/_synthetic_store.py`; it also runs
on the repository's `outputs`.

**Status.** Runs today. Its claim is redrawn from whatever study C table is on disk. On the
re-run's table (2026-09-16, 23:20 UTC; every row has its task result) both policy columns are
clean and enter the claim (SPEC §10.2; `outputs/studies/s6_shadow_rotation`, 45 of 45 rows, 0
contaminated). Before the re-run (the 16:02 UTC table), `sabr_linked` was excluded as
contaminated (15/15 rows).

---

## S7 — hedging

**Question.** How much does each hedging strategy leak when the world is not the pricing model,
and which delta regime sits closest to the minimum-variance delta?

**What it does** (runner `volsto.studies.catalogue.s7_hedging`; SPEC §10.2, §8.2). S7 reads
artefacts only: M8b tables A, B and D, and the discriminator verdict. It reports:

- the model-mismatch reserve, i.e. the leakage minus the same-world leakage, per world and
  product, in the §8.2 units and the desk sign;
- the historical world, carried as skipped with the verdict's reason (today: "surface artefact");
- the study-D headline, quoted and checked against the table: the regimes' distance to the common
  minimum-variance delta.

**Surface.** M8b tables A, B and D were computed on the SPX 2022-12-30 plain-SSVI snapshot (see
S5, "Surface").

**Run.**

```bash
volsto-study run configs/studies/catalogue/s7.yaml
volsto-study run configs/studies/catalogue/s7_fast.yaml --outputs <outputs root> --out <dir>
```

**Status.** Runs today.

---

## Backtest — rolling date-by-date marks and P&L

**Question** (as configured for 2022 H2). Day by day over 2022 H2, how does the marked book
move, which risks explain its P&L, and how do the marked SSR and the fitted parameters compare
with what was realised?

**What it does** (`volsto.studies.backtest`, console script `volsto-backtest`; SPEC §10.3). The
backtest has two stages.

**Stage 1**, `volsto-backtest run`, is the only M10 path that calibrates. For each trading date
it:

1. imports and fits the surface (eSSVI with the calendar repair);
2. runs the marking fit at (`ssr_target`, `skew_eps`), stage 3 off;
3. gets the leverage through the cache;
4. prices the book;
5. attributes the P&L of $(d-1, d]$ with the sticky-leverage `explain`;
6. publishes the date's outcome (`rows.parquet`, `fit.json`, `done.json`) as an immutable
   attempt `dates/<d>/attempts/<id>/` and points `dates/<d>/CURRENT` at it (SPEC §10.3,
   Integrity).

**The book.** A fixed book is struck on the first date: 3y autocall, 3y Phoenix, 1y cliquet,
12m VKO put, 1y KO variance swap and 1y variance swap. A rolling book strikes the same trades on
the first date of each month.

**Stage 2** is the study `configs/studies/catalogue/backtest_2022h2.yaml`. It renders from that
store and never calibrates. `configs/studies/catalogue/backtest_2022h2_poc.yaml` renders the
proof-of-concept window 2022-07-01..2022-08-05 of the same store: its backtest config
`configs/backtest/hdn_2022h2_poc.yaml` is `hdn_2022h2.yaml` with `dates.end` = 2022-08-05,
which is not hashed, so both share one config hash and one store. Stage 2 reports:

- the desk P&L (the desk short the book; the stored rows keep the holder's sign) per trade,
  bucket and month;
- the fitted parameters with stability flags;
- the realised SSR against the target and the first-order SSR;
- the VKO's mark against its realised state.

`study.md` states that 2022 H2 is a proof of concept: one regime, too short for a conclusion.
The multi-year run changes `dates.start`, `dates.end` and `data.root` in the config, and needs a
new store: `dates.start` is part of the config hash, so `run` refuses the old store. Set
`paths.out` and `paths.snapshots` (or pass `--out` and `--snapshots`).

**Run.**

```bash
volsto-backtest dry-run configs/backtest/hdn_2022h2.yaml        # the projected wall clock only
volsto-backtest run     configs/backtest/hdn_2022h2.yaml [--shard i/n] [--resume] \
    [--only-dates 2022-07-01..2022-08-05] [--limit N] [--no-calibrate] [--force] \
    [--out DIR] [--cache DIR] [--snapshots DIR]
volsto-backtest status  configs/backtest/hdn_2022h2.yaml
volsto-study run configs/studies/catalogue/backtest_2022h2.yaml      # stage 2, 127 dates
volsto-study run configs/studies/catalogue/backtest_2022h2_poc.yaml  # stage 2, the 25 PoC dates
```

With `VOLSTO_BACKTEST_REQUIRE_PATHS=1` in the environment, every `volsto-backtest` command that
lacks any of `--out`, `--cache`, `--snapshots` (or gives one blank) is refused (exit 2) before
anything is read or written. Set it for agents and scripted runs, so the config's `paths.*` are
never used. A blank path override is refused in any case (it would resolve to the current
directory). Every command `volsto-backtest` or the study prints (refusals, `status` hints,
stage-2 requirements, the command stored with a failed date) carries explicit `--out`, `--cache`
and `--snapshots`, shell-quoted, so it runs as printed under this variable and from paths with
spaces. A "write elsewhere" line names its new store `NEW_STORE` (replace it), with the
snapshots inside it.

**Sharding and resuming.**

- `--shard i/n` takes contiguous date blocks, because date $d$'s attribution needs date $d-1$'s
  leverage.
- `--resume` leaves a date alone when its verdict is `done` or `skipped`, or when it is `pending`
  on a date outside the selection. The verdict is read through the date's pointer
  `dates/<date>/CURRENT`: the pointed attempt's file hashes, then its dependency record
  recomputed (config hash, inputs, snapshot, leverage key and content, the verdicts of the dates
  it depends on). `volsto-backtest status` prints every verdict that is not `done` (all of them
  with `-v`).
- Under `--no-calibrate` a missing leverage exits 2 with the command.
- Every refusal (another config's store, dates computed under another config, a leverage
  missing under `--no-calibrate`, a `CURRENT` of a newer volsto-backtest) comes before anything
  is written, and prints the commands that would proceed.
- The per-date store defaults to `outputs/backtest/hdn_2022h2` (`paths.out`).
- The richer variant is `configs/backtest/hdn_2022h2_full.yaml`.

**Bring shards together.** Shards run on other machines write their own copies of the store,
the snapshots and the cache. Their dates are disjoint blocks. Their attempts are immutable and
named by content, so copies never collide. Copy each shard back with `rsync -a` and never
`--delete`, which would remove the other shards' dates, attempts and leverages. Paths below are
those of `hdn_2022h2.yaml`; `$SHARD` is the shard machine and `$R` its checkout.

```bash
# 1. the attempts first, then the pointers (a reader in between never sees a CURRENT whose
#    attempt has not arrived); in-flight writers' leftovers are not copied
rsync -a --partial --exclude CURRENT --exclude '.staging-*' --exclude '.CURRENT.tmp-*' \
  $SHARD:$R/outputs/backtest/hdn_2022h2/dates/ outputs/backtest/hdn_2022h2/dates/
rsync -a --partial --exclude '.staging-*' --exclude '.CURRENT.tmp-*' \
  $SHARD:$R/outputs/backtest/hdn_2022h2/dates/ outputs/backtest/hdn_2022h2/dates/
# 2. the snapshots and their import records
rsync -a --partial $SHARD:$R/outputs/backtest/hdn_2022h2/snapshots/ \
  outputs/backtest/hdn_2022h2/snapshots/
# 3. the leverages, without the cache manifest (merged below, never overwritten)
rsync -a --partial --exclude manifest.parquet --exclude manifest.parquet.lock \
  --exclude '.*.tmp*' $SHARD:$R/cache/ cache/
rsync -a $SHARD:$R/cache/manifest.parquet /tmp/shard_manifest.parquet
.venv/bin/python -c "
import pandas as pd
from volsto.calibration.cache import LeverageCache
before, added = LeverageCache('cache').merge_manifest(pd.read_parquet('/tmp/shard_manifest.parquet'))
print(before, '+', added, 'rows')"
# 4. check: every date done (a block's first date turns from pending to done once the date
#    before it has arrived)
volsto-backtest status configs/backtest/hdn_2022h2.yaml
```

Leave the store header (`backtest.json`) and `probe.json` alone: each shard's copy names the same
config hash. If both copies hold the same date (a date rerun on both machines), the pointer copied
last wins. `volsto-backtest run --resume` then applies the pointer rule, and `status` shows the
verdict. `LeverageCache.merge_manifest` adds only the keys the local manifest lacks, under the
cache's lock, and publishes atomically.

**Fast mode.** The toy backtest (`configs/backtest/hdn_2022h2_toy.yaml`) covers 5 dates across a
month end, $2\cdot10^4$ particles, a 1y horizon and 2000 paths. It **calibrates** 5 leverages;
SPEC §10.3 measured 163 s on an empty cache (142 s projected; the test fixture builds in
≈ 165 s). The test suite builds it in
`tests/_backtest_build.py::toy_backtest_build`, like this:

```bash
volsto-backtest run configs/backtest/hdn_2022h2_toy.yaml --out <root>/outputs/backtest/hdn_2022h2_toy \
    --cache <root>/cache --snapshots <root>/snapshots
volsto-study run configs/studies/catalogue/backtest_2022h2_fast.yaml \
    --outputs <root>/outputs --out <dir>
```

Both stages need the HistoricalData.net 2022 H2 sample under
`data/hdn_sample/options_sample_2022H2` (git-ignored). This recipe was not run for this page.

**Status.**

- `backtest_2022h2.yaml` exits 2 today: 25 of the 127 dates are computed (the proof-of-concept
  window 2022-07-01..2022-08-05, SPEC §10.3; `outputs/backtest/hdn_2022h2`). It prints
  `volsto-backtest run configs/backtest/hdn_2022h2.yaml --only-dates <102 dates> --resume`.
- `backtest_2022h2_poc.yaml` runs today on those 25 dates: 2453 numbers, 22 tables (the trades,
  inception, P&L and monthly tables split by unit) and 6 figures, about 11 s, recalibrated: no
  (rendered into a scratch directory on 2026-09-17). Its 25 dates carry record version 1, so the
  study states that their own leverage's numbers are not verified (key and completeness only)
  and the manifest lists them (`backtest_legacy_unverified`). The committed render in
  `outputs/studies/backtest_2022h2_poc` (1698 numbers, 14 tables; SPEC §10.3) predates the
  per-unit split. The PoC's stage 1 took 3.30 h wall clock against 10 248 s projected, with 25
  calibrations at 8·10⁵ particles.
- Run `dry-run` for the current projection before starting; it depends on a timing probe and
  moves between runs. SPEC §10.3 projects the shipped config (`hdn_2022h2.yaml`: ladders on 5
  pillars, rolling book priced at inception only, 127 dates) at 18.35 h in one process, 9.21 h
  on 2 shards and 4.71 h on 4. The configuration first written (ladders on 9 pillars, rolling
  book marked daily, no ξ₀ memo) projected 25.9 h.
