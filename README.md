## Run

From `src/`, using the existing Python environment. CP requires IBM CP Optimizer.

```sh
rtk bash run_solvers.sh --set I,II --solver maxsat_cpm2_solver.py,cp_solver.py
rtk bash run_solvers.sh --set II --solver all
```

`--solver` accepts filenames or `sat_cpm2,sat_cpm2_res_uni,maxsat_cpm2,maxsat_cpm2_res_uni,cp,milp,gurobi`.
`--set I`, `II`, or `I,II` selects datasets; default is all 51 files across both sets.
Defaults: sequential runs, `--time-limit 120`, `--memory-limit-mb 4096`, `--cp-workers 1`.
Tables, CSV, logs, and schedules: `benchmark_results/solver_comparison/<timestamp>/`.

## Papers

- Bofill et al. (2022). [The Sample Analysis Machine Scheduling Problem: Definition and comparison of exact solving approaches](https://doi.org/10.1016/j.cor.2022.105730).
- Coelho and Vanhoucke (2026). [Comparing and extending satisfiability solution methods for the resource-constrained project scheduling problem](https://doi.org/10.1016/j.cor.2025.107308).
