# Ising MCMC + Jerrum-Sinclair FPRAS

Ferromagnetic Ising on random 3-regular graphs.  Three estimators of `log Z(β,h)` compared on the same grid: exact (brute force, n≤20), MCMC + thermodynamic integration, and the Jerrum-Sinclair 1990 FPRAS.

## Files

| | |
|---|---|
| `ising.py` | single-site spin chain (Metropolis / Glauber × ground / uniform init) |
| `subgraphs.py` | JS 1990 subgraphs-world chain + `estimate_log_Z`, with tunable `step_n_mult` (1 = paper's 1/n schedule, larger = finer) |
| `simulate.py` | runs spin chains → `data/traces.csv` + `data/exact.csv` + `data/graphs/*.json`.  Incremental. |
| `run_thermo_integration.py` | post-processes `traces.csv` by trapezoidal β-integration of ⟨E⟩ → `data/log_z_mcmc.csv` (log Z from the same spin chains) |
| `run_log_z_experiment.py` | one-shot FPRAS + exact at every (graph, h, β) → `data/log_z.csv` (used as the "exact reference" lookup in plot) |
| `run_log_z_budget.py` | FPRAS sweep over `samples_per_segment ∈ {100,300,1000,3000,10000}` for all graphs, plus `step_n_mult ∈ {1,4,20}` for `n16_graph0` → `data/log_z_budget.csv` |
| `plot.py` | reads CSVs → `convergence_*.png` + `convergence.html` |
| `test_subgraphs.py` | unit tests for the subgraphs chain |

`run_js_fpras.py` is an older standalone sweep; superseded by `run_log_z_budget.py` but kept around.

## Grid

- **β** = `[0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 5.0]`
- **h** = `[0.0, 0.1, 0.2, 0.5, 1.0]`
- **Graphs** = 13 random 3-regular at n=16 (exact tractable), 3 at n=40

## CSVs (`data/`)

| file | one row per | written by |
|---|---|---|
| `traces.csv` | (graph, h, β, init, dyn, recorded step) | `simulate.py` |
| `exact.csv` | (n≤20 graph, h, β) | `simulate.py` |
| `log_z.csv` | (graph, h, β, method ∈ {exact, js_fpras}) | `run_log_z_experiment.py` |
| `log_z_mcmc.csv` | (graph, h, β, init, dyn, step) | `run_thermo_integration.py` |
| `log_z_budget.csv` | (graph, h, β, step_n_mult, samples_per_segment) | `run_log_z_budget.py` |
| `graphs/n{N}_graph{G}.json` | one per graph | `simulate.py` |

## `convergence.html` — interactive views

Top controls: graph radio, View radio (3), β / init / dyn filters.

- **energy convergence** — running ⟨E⟩(step) per (β, init, dyn).  For n=16 the y axis is relative error vs exact ⟨E⟩.
- **log Z (MCMC thermo)** — same chains, thermodynamic integration of ⟨E⟩ over β → log Z(step) per (β, init, dyn).
- **log Z (JS FPRAS budget)** — one independent FPRAS run per `samples_per_segment`, marker per run; `n16_graph0` also shows finer-schedule curves (dashed = 1/(4n), dotted = 1/(20n)).  Schedule-length table below.

## Typical workflow

```bash
python3 simulate.py                  # heavy, incremental
python3 run_thermo_integration.py    # fast
python3 run_log_z_experiment.py      # ~3 min
python3 run_log_z_budget.py          # ~10 min
python3 plot.py                      # fast
open convergence.html
```
