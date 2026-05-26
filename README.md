# Ising MCMC + Jerrum-Sinclair + Taylor

Ferromagnetic Ising on random 3-regular graphs. Four ways to estimate
`log Z(β, h)`: exact (brute force, n ≤ 20), MCMC + thermodynamic
integration, Jerrum-Sinclair 1990 FPRAS, and the Barvinok / Patel-Regts
deterministic Taylor truncation.

## Pipeline

```bash
python3 run_mcmc.py                  # spin chains  -> traces.csv + exact.csv
python3 run_thermo_integration.py    # traces.csv   -> log_z_mcmc.csv
python3 run_log_z_js.py              # JS+exact     -> log_z.csv + log_z_traces.csv
python3 run_log_z_js_sweep.py        # JS sweep     -> log_z_js_sweep.csv
python3 taylor/run_log_z_taylor.py   # Taylor       -> log_z_taylor.csv
python3 plot.py                      # everything   -> convergence.html
```

All producers are incremental — they skip cells already present in their
output CSV. Delete the CSV to force a clean re-run.

The Taylor producer computes the polynomial coefficients once per
`(graph, h, β, method)` up to the largest m in the grid, then reuses
them for every smaller m; `runtime_s` per row is the cumulative wall
time to reach that truncation order.

## Code

| | |
|---|---|
| `ising.py` | single-site spin chain (Metropolis / Glauber × ground / uniform init) |
| `subgraphs.py` | JS 1990 subgraphs chain + `estimate_log_Z`, with `step_n_mult` knob |
| `taylor/` | Barvinok / Patel-Regts coefficient extraction + LSS variable-change adapter |
| `plot.py` | reads CSVs, writes HTML + PNGs |

## Grid

- β = `[0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 5.0]`
- h = `[0.0, 0.1, 0.2, 0.5, 1.0]`
- Graphs: configured per producer (`GRAPHS_TO_RUN` set or `SIZES_SEEDS_COUNTS`).

## HTML views

- **energy** — running ⟨E⟩(step); n ≤ 20 shows relative error vs exact.
- **log Z (MCMC thermo)** — same chains, β-integrated → log Z.
- **log Z (Jerrum-Sinclair)** — one independent FPRAS run per
  `samples_per_segment`, with step-size refinements `step_n_mult ∈ {1, 4, 20}`.
- **log Z (Taylor)** — Barvinok truncation at orders m ∈ {2,4,6,8,10}
  (naive backend) or {2,4,6} (insects, n > 20).
- **log Z vs wall time** — all three estimators on a measured-time x-axis.
