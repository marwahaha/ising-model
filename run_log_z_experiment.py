"""Compute log Z(beta, h) on every graph under data/graphs/, at the same
(beta, h) grid as simulate.py.  For each (graph, h, beta) we run the
Jerrum-Sinclair FPRAS and record the convergence of the running estimate
as a function of total chain steps so we can plot accuracy vs work.

Methods
  - "exact"     : brute-force spin enumeration (only when n <= 20).
                  Vectorised; serves as ground truth.
  - "js_fpras"  : Jerrum-Sinclair 1990 subgraphs-world FPRAS, recorded at
                  num_log_samples log-spaced step counts.

Outputs (both under data/):
  log_z.csv          one row per (graph, h, beta, method) -- final estimate.
                     Columns: graph_id, n, h, beta, method, log_Z, runtime_s
  log_z_traces.csv   one row per (graph, h, beta, recorded step) -- the
                     FPRAS estimate as it converges.
                     Columns: graph_id, n, h, beta, step, log_Z_running
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import List, Tuple

from subgraphs import (
    estimate_log_Z_trace as js_trace,
    exact_log_Z,
)


BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
         1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 5.0]
H_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0]

# FPRAS hyperparameters.  Each schedule segment is a (burnin, sample) pair
# of inner-chain steps; total chain work per call is
#   |schedule| * (JS_BURNIN + JS_SAMPLES)
# where |schedule| <= n.  So for n=16 the FPRAS does up to ~208k inner steps;
# for n=40 up to ~520k.
JS_BURNIN = 3_000
JS_SAMPLES = 10_000
NUM_LOG_SAMPLES = 60

EXACT_MAX_N = 20

DATA_DIR = "data"
GRAPHS_SUBDIR = os.path.join(DATA_DIR, "graphs")
OUT_FINAL_CSV = os.path.join(DATA_DIR, "log_z.csv")
OUT_TRACE_CSV = os.path.join(DATA_DIR, "log_z_traces.csv")


def load_graph(graph_id: str) -> Tuple[List[Tuple[int, int]], int]:
    with open(os.path.join(GRAPHS_SUBDIR, f"{graph_id}.json")) as f:
        gj = json.load(f)
    return [tuple(e) for e in gj["edges"]], gj["n"]


def _sort_key(graph_id: str) -> Tuple[int, int]:
    n_part, g_part = graph_id.split("_")
    return int(n_part[1:]), int(g_part.replace("graph", ""))


def main():
    graph_ids = sorted(
        (fn[:-len(".json")] for fn in os.listdir(GRAPHS_SUBDIR)
         if fn.endswith(".json")),
        key=_sort_key,
    )
    print(f"Found {len(graph_ids)} graphs: {graph_ids}")

    fout = open(OUT_FINAL_CSV, "w", newline="")
    ftrace = open(OUT_TRACE_CSV, "w", newline="")
    try:
        w_final = csv.writer(fout)
        w_trace = csv.writer(ftrace)
        w_final.writerow(["graph_id", "n", "h", "beta", "method",
                          "log_Z", "runtime_s"])
        w_trace.writerow(["graph_id", "n", "h", "beta", "step",
                          "log_Z_running"])

        for graph_id in graph_ids:
            edges, n = load_graph(graph_id)
            print(f"\n=== {graph_id}  (n={n}, |E|={len(edges)}) ===")
            for h in H_VALUES:
                for beta in BETAS:
                    if n <= EXACT_MAX_N:
                        t0 = time.time()
                        lz_exact = exact_log_Z(edges, n, beta, h)
                        dt = time.time() - t0
                        w_final.writerow([graph_id, n, h, beta, "exact",
                                          f"{lz_exact:.6f}", f"{dt:.4f}"])
                        exact_str = f"exact={lz_exact:>9.4f}"
                    else:
                        lz_exact = None
                        exact_str = "exact=  --     "

                    seed = abs(hash((graph_id, beta, h, "js"))) % (1 << 32)
                    rng = random.Random(seed)
                    t0 = time.time()
                    lz_js, trace = js_trace(
                        edges, n, beta, h,
                        burnin=JS_BURNIN, samples_per_step=JS_SAMPLES,
                        num_log_samples=NUM_LOG_SAMPLES,
                        rng=rng,
                    )
                    dt = time.time() - t0
                    w_final.writerow([graph_id, n, h, beta, "js_fpras",
                                      f"{lz_js:.6f}", f"{dt:.4f}"])
                    for step, lz_run in trace:
                        w_trace.writerow([graph_id, n, h, beta,
                                          step, f"{lz_run:.6f}"])
                    print(f"  h={h:.1f}  beta={beta:>4.1f}   "
                          f"{exact_str}   js={lz_js:>9.4f}   "
                          f"trace_pts={len(trace):3d}   t={dt:5.2f}s")
                fout.flush()
                ftrace.flush()
    finally:
        fout.close()
        ftrace.close()
    print(f"\nWrote {OUT_FINAL_CSV}")
    print(f"Wrote {OUT_TRACE_CSV}")


if __name__ == "__main__":
    main()
