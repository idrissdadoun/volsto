# volsto documentation

| Page | What it covers |
|---|---|
| [methodology.md](methodology.md) | The model, the leverage calibration, the two-factor marking calibration (SABR break-evens, the two minimisations, the SSR and `skew_eps` dials), the conventions, and the known first-order limits of the break-even engine |
| [studies.md](studies.md) | The study runner and the catalogue S1–S7 plus the rolling backtest: the question each answers, how to run it, what runs today and which precompute each waits on |
| [vm_grid_run.md](vm_grid_run.md) | Running the default grid on a rented VM, sizing, and bringing the store and cache back |
| [../README.md](../README.md) | Install and quickstart (precompute a toy grid, run S1, open the viewer) |
| [../SPEC.md](../SPEC.md) | The authoritative design record, with every measured number |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | The standing rules the project is built under |

## Reference book

Place Bergomi, *Stochastic Volatility Modeling* (CRC Press, 2016) here as
`Stochastic_Volatility_Modeling.pdf`. It is git-ignored: copyrighted, local only.
SPEC.md §3.3, §4.1 and §4.4 cite its equation numbers.
