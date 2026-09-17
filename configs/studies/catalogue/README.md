# Study catalogue configs

One YAML per study of the M10 catalogue (`s1.yaml` … `s7.yaml`) and a CI-fast variant of each
(`s1_fast.yaml` … `s7_fast.yaml`), plus:

- `s1_grid.yaml`: S1 over the default grid's whole 1F sweep (exits 2 until that grid is
  precomputed);
- `s3_headline.yaml`: S3 on the cached headline models only;
- `backtest_2022h2.yaml`, `backtest_2022h2_poc.yaml` and `backtest_2022h2_fast.yaml`: stage 2 of
  the rolling backtest (runner `volsto.studies.backtest`, not a module of this directory). They
  read the per-date store that `volsto-backtest run configs/backtest/hdn_2022h2.yaml` writes;
  `_poc` renders the 25-date proof-of-concept window 2022-07-01..2022-08-05 of the same store,
  and `_fast` renders the toy backtest (SPEC §10.3).

Each is run by

```
volsto-study run configs/studies/catalogue/<id>.yaml [--out DIR] [--fast] [--grid G] [--store S]
                 [--cache C] [--outputs O] [--set key.sub=value ...] [--profile] [--no-latex-check]
```

and paired with a runner module under `volsto/studies/catalogue/` (the backtest configs with
`volsto.studies.backtest`). `volsto-study list` prints
every config here with its question. The runner's docstring (`volsto/studies/runner.py`) is the
reference; this file restates the schema.

## Schema (strict: an unknown key or a missing key is an error)

```yaml
name: s1_forward_vol            # [A-Za-z0-9_-]+; default output dir outputs/studies/<name>
runner: volsto.studies.catalogue.s1_forward_vol   # the runner module (dotted name)
question: "How do forward vol and cliquet prices move with vol-of-vol at a fixed smile?"
                                # optional; one line; defaults to the module's QUESTION
grid: configs/grids/default.yaml   # a volsto-precompute grid, or null
store: outputs/store            # M9 results store root
cache: cache                    # leverage cache root
outputs: outputs                # root of the M7 / M8b artefacts
mode: full                      # full | fast
seeds: {pricing: 2024}          # name -> non-negative integer; ctx.seed(name); a missing seed
                                # is an error
params: {...}                   # study-specific, plain YAML data
```

- The YAML is read strictly: a duplicated key is an error; `1e-3` and `2e4` are floats (YAML
  1.2); only `true` / `false` are booleans (`yes` stays a string); `.nan`, `.inf` and, inside
  `params`, the strings `nan` / `inf` are errors.
- Paths resolve against the repository root (the directory holding `pyproject.toml`), not the
  current directory. The CLI path flags are resolved against the current directory.
- `params`: every key the module lists in `REQUIRED_PARAMS` must be present; a key it lists in
  neither `REQUIRED_PARAMS` nor `OPTIONAL_PARAMS` is an error (a typo is never ignored). There
  are no silent defaults: the module's `validate_params(params)` (optional) checks the values.
- `--fast` sets `mode: fast`; `--set a.b=value` sets `params.a.b` to the value parsed the
  same way (an empty value is an error).
- A `point` requirement needs a grid (`grid: null` with a point requirement is an error). The
  printed `volsto-precompute` lines name the configured grid and run as printed from the current
  directory.
- The narrative (`narrative(results)`) is markdown. Write math as `\(...\)`; a `$` is a
  literal dollar sign. A fenced code block must not contain `\end{verbatim}`.

## Fast variants

`<id>_fast.yaml` has `mode: fast` and small budgets (paths, strikes, grid points). The tests
run each against a sanctioned session fixture, passing `--grid/--store/--cache/--outputs`:

- `s1_fast` … `s4_fast`: the toy precompute of `tests/conftest.py::toy_build`
  (`configs/grids/toy.yaml`: 2·10⁴ particles, 1y horizon);
- `s5_fast`: `tests/conftest.py::toy_marking_build` (`configs/grids/toy_marking.yaml`: four
  marking fits on the placeholder surface, 2·10⁴ particles, 1y horizon);
- `s6_fast`, `s7_fast` (`grid: null`): the synthetic M7 / M8b outputs that
  `tests/_synthetic_store.py::make_synthetic_outputs` writes under `toy_build`'s outputs root
  (the toy precompute and grid are not read);
- `backtest_2022h2_fast`: the toy backtest store of `tests/_backtest_build.py::toy_backtest_build`,
  run by `tests/test_backtest.py` (the catalogue walker `tests/test_catalogue_s1_s4.py` skips it).

Fast mode checks the plumbing, not the numbers. A study whose question needs a product beyond
the toy horizon says so in `study.md`.

## What a run writes

`results.parquet` (every number with its stderr), `tables/*.tex`, `figures/*.pdf` and `.png`,
`study.md`, `study.tex` (and `study.pdf` when the LaTeX check runs), and `manifest.json`: the
git commit, the code tags, the cache keys of every calibration used, the grid, the particle
counts, the seeds, the path counts, the wall clock, `recalibrated` (true only if a calibration
started in the process or a child, as recorded by the guard's `VOLSTO_CALIBRATION_MARKERS`
markers), `calibration_refusals`, and `cache_changed_during_run` (new or rewritten entries under
the cache root; a warning, not a failure). A study never calibrates: `calibrate_leverage` refuses while a study runs (in every thread
and in child processes); a calibration that starts anyway, or a refusal the study caught, fails
the run (`recalibrated: true` / `calibration_refusals`). A cache that another process changed
during the run is reported as "cache changed during the run, not attributed" (a warning).
Tables longer than 25 rows are split into numbered parts (`volsto.studies.latex.split_table`,
which also groups by an `axes` key). A missing store point, leverage or artefact prints the exact
command that produces it, and the run exits with status 2 before computing anything. The LaTeX check fails a
run whose `study.tex` drops a glyph, has a float taller than a page, or an overfull box.

`volsto-study rerun <dir>` re-executes the stored config (never this YAML) and reports every
number that moved by more than 2 stderr (exit 1). `volsto-study render <dir>` rebuilds the
tables, figures and documents from `results.parquet`.
