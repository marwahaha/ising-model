"""Long Glauber-from-uniform reference run for graphs too large to
brute-force.

For n > 20 the exact <E> by 2^n enumeration is intractable, so the log Z
views have no ground truth.  Here we build a pseudo-ground-truth: run the
Glauber chain from a uniform start 5x longer than the production chains
(run_mcmc.py uses N_STEPS = 400k; this uses 2M), and take the tail mean of
the energy (discarding the first half as burn-in) as the reference
<E>(beta, h).  Thermodynamic integration of that <E> over the beta grid
then gives a reference log Z(beta, h), using the same trapezoidal scheme
as run_thermo_integration.py:

    log Z(beta, h) = n*log 2  -  integral_0^beta <E>(beta', h) dbeta'.

Outputs:
  data/reference_E.csv    : graph_id, n, h, beta, ref_E, n_steps, burnin
  data/reference_logz.csv : graph_id, n, h, beta, ref_log_Z

Incremental: skips (graph_id, h, beta) already present in reference_E.csv.
The log Z file is fully rebuilt from reference_E.csv on each run (cheap).
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from ising import IsingChain

BETAS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 5.0]
H_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0]
REF_GRAPHS = ["n30_graph0", "n30_graph1", "n40_graph0", "n40_graph1"]
N_STEPS = 5 * 400_000        # 5x the production chain length
BURNIN_FRAC = 0.5            # discard first half, average E over the tail

DATA_DIR = "data"
GRAPHS_SUBDIR = os.path.join(DATA_DIR, "graphs")
REF_E_CSV = os.path.join(DATA_DIR, "reference_E.csv")
REF_LOGZ_CSV = os.path.join(DATA_DIR, "reference_logz.csv")


def load_graph(graph_id: str) -> Tuple[List[Tuple[int, int]], int]:
    with open(os.path.join(GRAPHS_SUBDIR, f"{graph_id}.json")) as f:
        g = json.load(f)
    return [tuple(e) for e in g["edges"]], g["n"]


def nx_adj(n: int, edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    G: Dict[int, List[int]] = {i: [] for i in range(n)}
    for u, v in edges:
        G[u].append(v)
        G[v].append(u)
    return G


def tail_mean_energy(G: Dict[int, List[int]], beta: float, h: float,
                     seed: int) -> float:
    """Glauber from uniform for N_STEPS; mean energy over the post-burn-in
    tail.  Energy is tracked incrementally."""
    rng = random.Random(seed)
    chain = IsingChain(G, h=h, beta=beta, rng=rng, init="uniform",
                       dynamics="glauber")
    E = chain.energy()
    nodes = chain.nodes
    sigma = chain.sigma
    adj = chain.G
    e = math.e
    burnin = int(BURNIN_FRAC * N_STEPS)
    acc = 0.0
    for t in range(1, N_STEPS + 1):
        v = rng.choice(nodes)
        nb = sum(sigma[u] for u in adj[v])
        z = 2.0 * beta * (nb + h)
        if z >= 0:
            p_plus = 1.0 / (1.0 + pow(e, -z))
        else:
            ez = pow(e, z)
            p_plus = ez / (1.0 + ez)
        new = 1 if rng.random() < p_plus else -1
        if new != sigma[v]:
            E += 2.0 * sigma[v] * (nb + h)
            sigma[v] = new
        if t > burnin:
            acc += E
    return acc / (N_STEPS - burnin)


def _existing_ref_E(path: str) -> set:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.add((row["graph_id"], float(row["h"]), float(row["beta"])))
    return out


def rebuild_logz() -> None:
    """Thermo-integrate every (graph, h) in reference_E.csv into a reference
    log Z, written to reference_logz.csv (full rewrite)."""
    by_gh: Dict[Tuple[str, float], Dict[float, float]] = defaultdict(dict)
    n_per_graph: Dict[str, int] = {}
    with open(REF_E_CSV, newline="") as f:
        for row in csv.DictReader(f):
            gid = row["graph_id"]
            n_per_graph[gid] = int(row["n"])
            by_gh[(gid, float(row["h"]))][float(row["beta"])] = float(row["ref_E"])

    with open(REF_LOGZ_CSV, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["graph_id", "n", "h", "beta", "ref_log_Z"])
        for (gid, h), beta_to_E in sorted(by_gh.items()):
            n = n_per_graph[gid]
            pts = [(0.0, 0.0)] + sorted(beta_to_E.items())
            log_Z = n * math.log(2.0)
            for i in range(1, len(pts)):
                b_prev, e_prev = pts[i - 1]
                b_curr, e_curr = pts[i]
                log_Z -= 0.5 * (b_curr - b_prev) * (e_curr + e_prev)
                w.writerow([gid, n, h, b_curr, f"{log_Z:.6f}"])
    print(f"Wrote {REF_LOGZ_CSV}")


def main() -> None:
    done = _existing_ref_E(REF_E_CSV)
    write_header = not os.path.exists(REF_E_CSV)
    mode = "w" if write_header else "a"
    burnin = int(BURNIN_FRAC * N_STEPS)
    with open(REF_E_CSV, mode, newline="") as fout:
        w = csv.writer(fout)
        if write_header:
            w.writerow(["graph_id", "n", "h", "beta", "ref_E",
                        "n_steps", "burnin"])
        for graph_id in REF_GRAPHS:
            edges, n = load_graph(graph_id)
            G = nx_adj(n, edges)
            print(f"\n=== {graph_id} (n={n}, |E|={len(edges)}, "
                  f"N_STEPS={N_STEPS:,}) ===")
            for h in H_VALUES:
                for beta in BETAS:
                    if (graph_id, h, beta) in done:
                        continue
                    seed = abs(hash((graph_id, h, beta, "ref"))) % (1 << 32)
                    t0 = time.time()
                    ref_E = tail_mean_energy(G, beta, h, seed)
                    dt = time.time() - t0
                    w.writerow([graph_id, n, h, beta, f"{ref_E:.6f}",
                                N_STEPS, burnin])
                    print(f"  h={h:.1f}  beta={beta:>4.1f}  "
                          f"ref_E={ref_E:>10.4f}  t={dt:5.1f}s")
                fout.flush()
    rebuild_logz()
    print(f"\nWrote {REF_E_CSV}")


if __name__ == "__main__":
    main()
