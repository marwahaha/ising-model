"""Sweep samples_per_segment (and step_n_mult on the highlighted graph) to
study FPRAS convergence vs total chain work.

Two knobs:
  samples_per_segment  MCMC samples drawn per (mu_k -> mu_{k+1}) link.
                        Bigger => lower per-link variance.
  step_n_mult          1 => paper's 1/n step (n segments); larger refines
                        the schedule to 1/(step_n_mult * n) (more segments,
                        each with tighter per-link variance window).

Scope:
  - All graphs at step_n_mult = 1 (the paper's schedule).
  - On HIGHLIGHTED_GRAPHS, also run step_n_mult in {4, 20} so we can
    overlay step-size effects in the log Z view.

Output: data/log_z_budget.csv with columns
  graph_id, n, h, beta, step_n_mult,
  samples_per_segment, n_segments,
  burnin, total_steps, log_Z, runtime_s
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from typing import List, Tuple

from subgraphs import estimate_log_Z, schedule_mu


BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
         1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 5.0]
H_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0]

BURNIN = 3_000
SAMPLES_GRID = [100, 300, 1000, 3000, 10_000]
STEP_N_MULTS = [1, 4, 20]   # applied to every graph

DATA_DIR = "data"
GRAPHS_SUBDIR = os.path.join(DATA_DIR, "graphs")
OUT_CSV = os.path.join(DATA_DIR, "log_z_budget.csv")


def load_graph(graph_id: str) -> Tuple[List[Tuple[int, int]], int]:
    with open(os.path.join(GRAPHS_SUBDIR, f"{graph_id}.json")) as f:
        gj = json.load(f)
    return [tuple(e) for e in gj["edges"]], gj["n"]


def _sort_key(gid: str) -> Tuple[int, int]:
    n_part, g_part = gid.split("_")
    return int(n_part[1:]), int(g_part.replace("graph", ""))


def main():
    graph_ids = sorted(
        (fn[:-len(".json")] for fn in os.listdir(GRAPHS_SUBDIR)
         if fn.endswith(".json")),
        key=_sort_key,
    )
    with open(OUT_CSV, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["graph_id", "n", "h", "beta", "step_n_mult",
                    "samples_per_segment", "n_segments",
                    "burnin", "total_steps", "log_Z", "runtime_s"])
        for graph_id in graph_ids:
            edges, n = load_graph(graph_id)
            print(f"\n=== {graph_id}  (n={n}, |E|={len(edges)}, "
                  f"step_n_mults={STEP_N_MULTS}) ===")
            for h in H_VALUES:
                for beta in BETAS:
                    mu_target = math.tanh(beta * h)
                    for step_n_mult in STEP_N_MULTS:
                        n_segs = len(schedule_mu(
                            n, mu_target, step_n_mult=step_n_mult)) - 1
                        for m in SAMPLES_GRID:
                            seed = abs(hash((graph_id, h, beta,
                                             step_n_mult, m))) % (1 << 32)
                            rng = random.Random(seed)
                            t0 = time.time()
                            try:
                                log_Z = estimate_log_Z(
                                    edges, n, beta, h,
                                    burnin=BURNIN, samples_per_step=m,
                                    step_n_mult=step_n_mult, rng=rng,
                                )
                                log_Z_str = f"{log_Z:.6f}"
                            except RuntimeError:
                                log_Z_str = "nan"
                            dt = time.time() - t0
                            total_steps = n_segs * (BURNIN + m)
                            w.writerow([graph_id, n, h, beta, step_n_mult,
                                        m, n_segs, BURNIN,
                                        total_steps, log_Z_str,
                                        f"{dt:.4f}"])
                    print(f"  h={h:.1f}  beta={beta:>4.1f}  "
                          f"n_segs={[len(schedule_mu(n, mu_target, step_n_mult=s))-1 for s in STEP_N_MULTS]}")
                fout.flush()
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
