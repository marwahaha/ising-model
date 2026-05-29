# Ising MCMC + Jerrum-Sinclair + Taylor

Ferromagnetic Ising on random 3-regular graphs. Estimate `log Z(β, h)` five
ways and compare them in one interactive HTML:

- **exact** — brute force over all 2ⁿ spins (n ≤ 20 only).
- **MCMC thermo** — ⟨E⟩(β) from single-site chains, trapezoidally
  integrated: `log Z = n·log2 − ∫₀^β ⟨E⟩ dβ'`.
- **MCMC FEP** — same chains, telescoped by free-energy perturbation
  (Zwanzig): `log Z(β_{k+1}) = n·log2 + Σ_k log⟨e^{−Δβ·E}⟩_{β_k}`. Each ratio
  is exact in expectation, so unlike the trapezoidal estimate it carries no
  β-grid discretization bias (validated vs exact at n=16: ~2× lower error,
  biggest gains in the β≈0.5–1.8 transition).
- **MCMC FEP baseline** (`run_fep_dense.py`) — a higher-accuracy FEP run used
  as the `n > 20` reference for relative-error plots. Same telescoping recipe
  but on a dense β-ladder (Δβ=0.1, 50 segments from 0 to 5.0) with much
  longer chains per segment (2M post-burn-in samples), ground init aligned
  with sign(h), Glauber dynamics. Output: `data/log_z_fep_baseline.csv`.
  Validated at n=16 vs exact log Z: mean relative error **~2.7e-4** (max
  ~1.7e-3) across 200 anchors, ~5× more accurate than the coarse FEP it
  replaces as the reference.
- **Jerrum-Sinclair** — the 1990 subgraphs-world FPRAS, telescoping the field
  activity μ = tanh(βh) from the closed-form μ=1 end (`Z'(1) = Π_e(1+λ_e)`)
  *down* to the target.
- **Taylor** — Barvinok / Patel-Regts deterministic truncation of `log Z(λ)`
  as a power series in the edge activity (Lee-Yang ⇒ zeros on |λ|=1, so the
  series converges for |λ| < 1).
- **HTE (high-temperature expansion, h=0 only)** — cycle expansion:
  `log Z(β) ≈ |V|·log 2 + |E|·log cosh(β) + Σ_{k=3}^{K} f(k)·log(1+tanh(β)^k)`.
  Two variants are plotted: **asym** uses `f(k) = 2^k/(2k)` (asymptotic
  expected k-cycle count in random 3-regular graphs, graph-independent);
  **exact f(3,4,5)** uses true per-graph cycle counts for k=3,4,5 (direct
  enumeration, O(n·d^(k-1)) ≈ 90–340 µs for n=16–60) and asymptotic for
  k≥6. Plotted for K=3..n; diverges for β ≳ 0.5 once K is large enough
  that f(k) overwhelms tanh(β)^k. At small β the exact variant is 2–3
  orders of magnitude more accurate than asymptotic.

For n > 20 there is no tractable brute force, so the dense-ladder FEP
baseline (`data/log_z_fep_baseline.csv`, see "MCMC FEP baseline" below) is
used as the relative-error reference.

## Setup

```bash
pip install -r requirements.txt   # numpy, networkx, plotly, matplotlib (Python 3)
```

Plotly's JavaScript bundle is fetched from the CDN into `vendor/` on the first
`plot.py` run; everything else runs offline.

## Pipeline

```bash
python3 run_mcmc.py                  # spin chains   -> traces.csv + exact.csv (+ graphs)
python3 run_thermo_integration.py    # traces.csv    -> log_z_mcmc.csv
python3 run_fep.py                   # FEP telescope -> log_z_fep.csv
python3 run_fep_dense.py             # dense-ladder FEP (high-accuracy reference)
python3 merge_baseline.py            #   -> data/log_z_fep_baseline.csv  (+ validation)
python3 run_log_z_js.py              # JS + exact    -> log_z.csv + log_z_traces.csv
python3 run_log_z_js_sweep.py        # JS sweep      -> log_z_js_sweep.csv
python3 taylor/run_log_z_taylor.py   # Taylor        -> log_z_taylor.csv
python3 run_hte.py                   # HTE cycle exp -> log_z_hte.csv
python3 plot.py                      # everything    -> convergence.html + PNGs
```

`run_mcmc.py` also generates the graphs under `data/graphs/`, so run it
first. All producers are incremental — they skip cells already present in
their output CSV; delete the CSV to force a clean re-run.

The wall-time view converts step counts to seconds using microbenched
per-step rates from `run_microbench.py` (MCMC/JS) and `run_microbench_fep.py`
(FEP); run these on an idle machine and paste the printed dicts into
`plot.py`.

## Code

| | |
|---|---|
| `ising.py` | single-site spin chain (Metropolis / Glauber × ground / uniform init) |
| `run_fep.py` | FEP telescoping of log Z over the β-ladder, per (init, dynamics) |
| `run_fep_dense.py` | dense-ladder FEP run used as the n>20 reference (glauber + ground init) |
| `merge_baseline.py` | merge per-graph baseline partials + validate vs exact at n=16 |
| `subgraphs.py` | JS 1990 subgraphs chain + `estimate_log_Z` (`step_n_mult` knob) |
| `taylor/` | Barvinok / Patel-Regts coefficient extraction + activity-change adapter |
| `run_hte.py` | high-temperature (cycle) expansion at h=0, K = 3..n, with f(k) = 2^k/(2k) |
| `plot.py` | reads CSVs, writes the combined HTML + per-graph PNGs |

## Grid

- β = `[0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 5.0]`
- h = `[0.0, 0.1, 0.2, 0.5, 1.0]`
- Graphs: 3-regular, n ∈ {16, 30, 40, 50, 60}; configured per producer
  (`GRAPHS_TO_RUN` / `SIZES_SEEDS_COUNTS`).

## HTML views

The prebuilt `convergence.html` opens directly in a browser (or regenerate it
with `python3 plot.py`). Pick a graph and toggle β / init / dynamics. Each **log Z** view shows two
stacked sets — raw `log Ẑ` on top, relative error below (reference: exact for
n ≤ 20, FEP otherwise):

- **energy** — running ⟨E⟩(step).
- **log Z (MCMC thermo)** — trapezoidal β-integration of the chains' ⟨E⟩.
- **log Z (MCMC FEP)** — FEP telescoping; same Metropolis/Glauber ×
  low-temp/high-temp toggles as the energy view.
- **log Z (Jerrum-Sinclair)** — one FPRAS run per `samples_per_segment`, with
  step-size refinements `step_n_mult ∈ {1, 4, 20}`.
- **log Z (Taylor)** — truncation orders m ∈ {2,4,6,8,10} (naive, n ≤ 20) or
  {2,4,6} (insects, n > 20).
- **log Z (HTE, h=0)** — high-temperature expansion at h=0; two curves
  per β (asym / exact f(3,4,5)), x = truncation K from 3 to n; h ≠ 0
  panels are blank. Also drawn as two trace groups on the wall-time view,
  with the exact variant carrying its measured cycle-count setup cost.
- **log Z vs wall time** — all estimators on a measured wall-time x-axis.

Init labels: **low-temp** = ground (all-aligned) start, **high-temp** =
uniform (random) start. The MCMC step accounting includes burn-in; FEP and JS
both use 3,000 burn-in steps per segment.

## Tests

```bash
python3 test_subgraphs.py   # 6 checks of the JS chain + FPRAS (also runs under pytest)
```
