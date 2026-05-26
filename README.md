# Ising MCMC + Jerrum-Sinclair FPRAS

Ferromagnetic Ising on random 3-regular graphs. Three ways to estimate `log Z(β, h)`:
exact (brute force, n ≤ 20), MCMC + thermodynamic integration, and the
Jerrum-Sinclair 1990 FPRAS.

## Pipeline

```bash
python3 simulate.py                  # spin chains  -> traces.csv + exact.csv
python3 run_thermo_integration.py    # traces.csv   -> log_z_mcmc.csv
python3 run_log_z_experiment.py      # FPRAS+exact  -> log_z.csv + log_z_traces.csv
python3 run_log_z_budget.py          # FPRAS sweep  -> log_z_budget.csv
python3 plot.py                      # everything   -> convergence.html
```

`simulate.py` is incremental; delete `data/traces.csv` to force a re-run.

## Code

| | |
|---|---|
| `ising.py` | single-site spin chain (Metropolis / Glauber × ground / uniform init) |
| `subgraphs.py` | JS 1990 subgraphs chain + `estimate_log_Z`, with `step_n_mult` knob (1 = paper's 1/n schedule) |
| `plot.py` | reads CSVs, writes HTML + PNGs |

## Grid

- β = `[0.1, 0.2, …, 1.6, 1.8, 2.0, 2.5, 5.0]` (20 values)
- h = `[0.0, 0.1, 0.2, 0.5, 1.0]`
- 13 graphs at n=16 (exact tractable), 3 at n=40

## HTML views

Top controls: graph picker, view radio (4), β / init / dyn filters.

- **energy** — running ⟨E⟩(step); n=16 shows relative error vs exact.
- **log Z (MCMC thermo)** — same chains, β-integrated → log Z.
- **log Z (JS FPRAS)** — one independent FPRAS run per `samples_per_segment`;
  step-size refinements `step_n_mult ∈ {1, 4, 20}`.
- **log Z vs wall time** — MCMC and FPRAS overlaid on a measured µs/step axis
  (Metropolis 1.67, Glauber 1.90, JS 1.01 µs/step).
