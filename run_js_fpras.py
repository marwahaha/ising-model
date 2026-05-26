"""Run the Jerrum-Sinclair FPRAS on every n=16 graph we have under
data/graphs/, for the same (beta, h) grid as the spin-Metropolis convergence
experiments.

For each (graph, beta, h):
  - compute log Z exactly by spin enumeration (n=16: 2^16 = 65536 configs);
  - estimate log Z with the JS FPRAS at fixed (burnin, samples_per_step);
  - record both, with |error| and relative error.

Output: data/log_Z.csv with columns
  graph_id, n, h, beta, log_Z_exact, log_Z_js, abs_err, rel_err, runtime_s

NOTE: n=40 graphs are skipped here -- brute-force log Z is infeasible at that
size (2^40 ~= 10^12), so there is no ground-truth to compare against in the
same way.  The JS FPRAS would still run there; that is a separate experiment.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from typing import List, Tuple

from subgraphs import estimate_log_Z, exact_log_Z

BETAS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 5.0]
H_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0]
BURNIN = 3_000
SAMPLES_PER_STEP = 10_000

DATA_DIR = "data"
GRAPHS_SUBDIR = os.path.join(DATA_DIR, "graphs")
OUT_CSV = os.path.join(DATA_DIR, "log_Z.csv")


def load_graph(graph_id: str) -> Tuple[List[Tuple[int, int]], int]:
    with open(os.path.join(GRAPHS_SUBDIR, f"{graph_id}.json")) as f:
        gj = json.load(f)
    return [tuple(e) for e in gj["edges"]], gj["n"]


def main():
    graph_ids = sorted(
        fn[:-len(".json")] for fn in os.listdir(GRAPHS_SUBDIR)
        if fn.startswith("n16_") and fn.endswith(".json")
    )
    print(f"Found {len(graph_ids)} n=16 graphs: {graph_ids}")

    with open(OUT_CSV, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["graph_id", "n", "h", "beta",
                         "log_Z_exact", "log_Z_js", "abs_err", "rel_err",
                         "runtime_s"])
        for graph_id in graph_ids:
            edges, n = load_graph(graph_id)
            print(f"\n=== {graph_id}  (n={n}, |E|={len(edges)}) ===")
            for h in H_VALUES:
                for beta in BETAS:
                    log_Z_exact = exact_log_Z(edges, n, beta, h)
                    seed = abs(hash((graph_id, beta, h))) % (1 << 32)
                    rng = random.Random(seed)
                    t0 = time.time()
                    log_Z_js = estimate_log_Z(
                        edges, n, beta, h,
                        burnin=BURNIN, samples_per_step=SAMPLES_PER_STEP,
                        rng=rng,
                    )
                    dt = time.time() - t0
                    err = abs(log_Z_js - log_Z_exact)
                    rel = err / abs(log_Z_exact) if log_Z_exact != 0 else err
                    print(f"  h={h:.1f}  beta={beta:>4.1f}   "
                          f"exact={log_Z_exact:>9.4f}   js={log_Z_js:>9.4f}   "
                          f"|Δ|={err:.4f}   rel={rel:6.2%}   t={dt:5.2f}s")
                    writer.writerow([graph_id, n, h, beta,
                                     f"{log_Z_exact:.6f}",
                                     f"{log_Z_js:.6f}",
                                     f"{err:.6f}",
                                     f"{rel:.6f}",
                                     f"{dt:.3f}"])
                    fout.flush()
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
