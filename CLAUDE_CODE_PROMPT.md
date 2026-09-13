You are building `volsto`, a Python/numba stochastic-volatility pricing library for
single-underlying light-exotic and exotic parameter studies. The complete specification
(v1.1) is in ./SPEC.md — read it in full before writing any code and treat it as the
source of truth for layout, interfaces, equations, tests and milestone order. The
reference text is ./docs/Stochastic_Volatility_Modeling.pdf (Bergomi); wherever SPEC.md
cites a book equation number, the docstring of the implementing function must cite it too.

Context: the library replaces per-study re-implementations of an LSV Monte Carlo
engine (two-factor lognormal Bergomi forward-variance kernel, particle-method leverage
calibrated to an SSVI surface). It must reproduce the earlier forward-start / capped
cliquet / FVA studies as regression tests and support Streamlit viewers over a
precomputed parameter cache. Multi-asset is out of scope for v1 but PathSet must be
designed so a second underlying can be added later without changing product code.

Work milestone by milestone (SPEC.md §12), starting with M1. For each milestone:
1. Restate in a few lines what you will build and which SPEC sections apply.
2. Implement it, then the tests listed for it in §10; run them; do not move on until
   they are green (fast tests by default; mark full-size ones `slow`).
3. Every implemented equation gets a docstring stating the formula, its source (book
   equation number, or "derived" for the few the spec flags as derived), and which
   test checks it.
4. Stop and report at the end of each milestone with: what was built, test results,
   any deviation from SPEC.md and why, and open questions. Wait for my go-ahead.

Rules:
- Never return a Monte Carlo number without its standard error.
- All model parameters come from explicit validated YAML configs; no silent defaults.
- Numba kernels are pure array functions; validation lives in Python wrappers.
- Reproducibility: same (seed, path, step, brownian) always gives the same normal, so
  common-random-number bumps are exact.
- If a formula in SPEC.md fails its numerical test, do not patch the test — report the
  discrepancy with the numbers and your proposed correction.
- black, ruff, mypy strict on volsto/; pytest; pyproject.toml; README quickstart.

For milestone M4's regression tests you will need the original study archive
(SSVI parameters, seeds, result tables). If ./study_archive/ is absent when you reach
M4, implement the tests against a placeholder config and flag them as pending.

The PDF in docs/ is git-ignored on purpose (copyrighted); read it from disk, never commit it.

Begin with M1.
