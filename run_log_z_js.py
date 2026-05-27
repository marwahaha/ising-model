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


BETAS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 5.0]
H_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0]
GRAPHS_TO_RUN = {"n16_graph0", "n16_graph1", "n16_graph2", "n16_graph3",
                 "n30_graph0", "n30_graph1", "n40_graph0", "n40_graph1",
                 "n50_graph0", "n50_graph1",
                 "n60_graph0", "n60_graph1"}

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


def _existing_cells(path: str) -> set:
    """Set of (graph_id, h, beta, method) tuples already in OUT_FINAL_CSV."""
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.add((row["graph_id"], float(row["h"]),
                     float(row["beta"]), row["method"]))
    return out


def main():
    graph_ids = sorted(
        (fn[:-len(".json")] for fn in os.listdir(GRAPHS_SUBDIR)
         if fn.endswith(".json")),
        key=_sort_key,
    )
    graph_ids = [g for g in graph_ids if g in GRAPHS_TO_RUN]
    print(f"Found {len(graph_ids)} graphs: {graph_ids}")

    done = _existing_cells(OUT_FINAL_CSV)
    write_header = not os.path.exists(OUT_FINAL_CSV)
    final_mode = "w" if write_header else "a"
    trace_mode = "w" if write_header else "a"
    fout = open(OUT_FINAL_CSV, final_mode, newline="")
    ftrace = open(OUT_TRACE_CSV, trace_mode, newline="")
    try:
        w_final = csv.writer(fout)
        w_trace = csv.writer(ftrace)
        if write_header:
            w_final.writerow(["graph_id", "n", "h", "beta", "method",
                              "log_Z", "runtime_s"])
            w_trace.writerow(["graph_id", "n", "h", "beta", "step",
                              "log_Z_running"])

        for graph_id in graph_ids:
            edges, n = load_graph(graph_id)
            print(f"\n=== {graph_id}  (n={n}, |E|={len(edges)}) ===")
            for h in H_VALUES:
                for beta in BETAS:
                    if n <= EXACT_MAX_N and (graph_id, h, beta, "exact") not in done:
                        t0 = time.time()
                        lz_exact = exact_log_Z(edges, n, beta, h)
                        dt = time.time() - t0
                        w_final.writerow([graph_id, n, h, beta, "exact",
                                          f"{lz_exact:.6f}", f"{dt:.4f}"])
                        exact_str = f"exact={lz_exact:>9.4f}"
                    else:
                        lz_exact = None
                        exact_str = "exact=  cached  "

                    if (graph_id, h, beta, "js_fpras") in done:
                        print(f"  h={h:.1f}  beta={beta:>4.1f}   "
                              f"{exact_str}   js=cached")
                        continue

                    seed = abs(hash((graph_id, beta, h, "js"))) % (1 << 32)
                    rng = random.Random(seed)
                    t0 = time.time()
                    try:
                        lz_js, trace = js_trace(
                            edges, n, beta, h,
                            burnin=JS_BURNIN, samples_per_step=JS_SAMPLES,
                            num_log_samples=NUM_LOG_SAMPLES,
                            rng=rng,
                        )
                        lz_str = f"{lz_js:.6f}"
                    except RuntimeError as exc:
                        # Hard h=0 / large-n corner: the final mu->0 ratio
                        # (even-subgraph fraction) can underflow to 0 in finite
                        # samples.  Record nan rather than crash (the sweep
                        # does the same).
                        lz_js, trace, lz_str = float("nan"), [], "nan"
                        print(f"   ! h={h:.1f} beta={beta:.1f}: {exc}")
                    dt = time.time() - t0
                    w_final.writerow([graph_id, n, h, beta, "js_fpras",
                                      lz_str, f"{dt:.4f}"])
                    for step, lz_run in trace:
                        w_trace.writerow([graph_id, n, h, beta,
                                          step, f"{lz_run:.6f}"])
                    if trace or lz_str != "nan":
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
