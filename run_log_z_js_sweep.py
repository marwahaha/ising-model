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

Output: data/log_z_js_sweep.csv with columns
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


BETAS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 5.0]
H_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0]
GRAPHS_TO_RUN = {"n16_graph0", "n16_graph1", "n16_graph2", "n16_graph3",
                 "n30_graph0", "n30_graph1", "n40_graph0", "n40_graph1",
                 "n50_graph0", "n50_graph1",
                 "n60_graph0", "n60_graph1"}

BURNIN = 3_000
SAMPLES_GRID = [100, 300, 1000, 3000, 10_000]
STEP_N_MULTS = [1, 4, 20]        # default: full step-size sweep
STEP_N_MULTS_LARGE = [1]         # n >= LARGE_N: only the paper's 1/n schedule
LARGE_N = 40                     # finer schedules get too slow at this size

DATA_DIR = "data"
GRAPHS_SUBDIR = os.path.join(DATA_DIR, "graphs")
OUT_CSV = os.path.join(DATA_DIR, "log_z_js_sweep.csv")


def load_graph(graph_id: str) -> Tuple[List[Tuple[int, int]], int]:
    with open(os.path.join(GRAPHS_SUBDIR, f"{graph_id}.json")) as f:
        gj = json.load(f)
    return [tuple(e) for e in gj["edges"]], gj["n"]


def _sort_key(gid: str) -> Tuple[int, int]:
    n_part, g_part = gid.split("_")
    return int(n_part[1:]), int(g_part.replace("graph", ""))


def _existing_cells(path: str) -> set:
    """Set of (graph_id, h, beta, step_n_mult, samples_per_segment) already
    in the CSV.  Skipped on subsequent runs."""
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.add((row["graph_id"], float(row["h"]), float(row["beta"]),
                     int(row["step_n_mult"]), int(row["samples_per_segment"])))
    return out


def main():
    graph_ids = sorted(
        (fn[:-len(".json")] for fn in os.listdir(GRAPHS_SUBDIR)
         if fn.endswith(".json")),
        key=_sort_key,
    )
    graph_ids = [g for g in graph_ids if g in GRAPHS_TO_RUN]
    done = _existing_cells(OUT_CSV)
    write_header = not os.path.exists(OUT_CSV)
    mode = "w" if write_header else "a"
    with open(OUT_CSV, mode, newline="") as fout:
        w = csv.writer(fout)
        if write_header:
            w.writerow(["graph_id", "n", "h", "beta", "step_n_mult",
                        "samples_per_segment", "n_segments",
                        "burnin", "total_steps", "log_Z", "runtime_s"])
        for graph_id in graph_ids:
            edges, n = load_graph(graph_id)
            step_mults = STEP_N_MULTS_LARGE if n >= LARGE_N else STEP_N_MULTS
            print(f"\n=== {graph_id}  (n={n}, |E|={len(edges)}, "
                  f"step_n_mults={step_mults}) ===")
            for h in H_VALUES:
                for beta in BETAS:
                    mu_target = math.tanh(beta * h)
                    for step_n_mult in step_mults:
                        n_segs = len(schedule_mu(
                            n, mu_target, step_n_mult=step_n_mult)) - 1
                        for m in SAMPLES_GRID:
                            if (graph_id, h, beta, step_n_mult, m) in done:
                                continue
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
                          f"n_segs={[len(schedule_mu(n, mu_target, step_n_mult=s))-1 for s in step_mults]}")
                fout.flush()
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
