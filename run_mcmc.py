"""Run the Ising single-site MCMC and save running-mean energy traces to CSV.

This script is the heavy step.  It writes everything needed to produce plots
later to the `data/` directory:

  - data/traces.csv  : long-form running-mean energy trace.  Columns:
        graph_id, n, h, beta, init, dynamics, step, running_mean_E
  - data/exact.csv   : exact <E>_{beta,h} for n <= 20 graphs, by brute force.
        Columns: graph_id, n, h, beta, exact_E
  - data/graphs/n{N}_graph{G}.json : adjacency-list dump of each graph used,
        so plotting can draw the graph itself.

Incremental:  if traces.csv already exists, the script reads its graph_id
column and SKIPS any (n, g_idx) whose graph_id is already present.  New
rows are appended.  To force a full re-run, delete the data/ directory
first.

Configuration is in the constants at the top of this file.

Initial distributions ("init"):
  - "ground"  : all spins aligned with sign(h) (uniform tie-break at h=0)
  - "uniform" : uniform over all 2^|V| spin configurations
Dynamics rules ("dynamics"):
  - "metropolis" : propose flipping sigma_v; accept w.p. min(1, exp(-beta dE))
  - "glauber"    : resample sigma_v from its conditional given neighbours
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from typing import List, Tuple

import networkx as nx
import numpy as np

from ising import IsingChain, Graph


BETAS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 5.0]
H_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0]
INITS = ["ground", "uniform"]
DYNAMICS = ["metropolis", "glauber"]
N_STEPS = 400_000
NUM_LOG_SAMPLES = 120
# (n, base_seed_for_graphs, num_graphs).  Mix of n=16, n=30, n=40.
# n=40 reuses the seed=200 graphs already on disk.
SIZES_SEEDS_COUNTS = [(16, 100, 4), (30, 300, 2), (40, 200, 2),
                      (50, 250, 2), (60, 350, 2)]

DATA_DIR = "data"
GRAPHS_SUBDIR = os.path.join(DATA_DIR, "graphs")
TRACES_CSV = os.path.join(DATA_DIR, "traces.csv")
EXACT_CSV = os.path.join(DATA_DIR, "exact.csv")


# ----- exact <E> by brute force -----

def exact_expected_energy(edges: List[Tuple[int, int]], n: int,
                          beta: float, h: float = 0.0) -> float:
    """Return <H>_{beta,h} by summing the Boltzmann measure over all 2^n
    spin configurations.  Numerically stable via log-sum-exp.  Tractable
    only for n <= ~24.
    """
    idx = np.arange(1 << n, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(n)[None, :]) & 1
    spins = 2 * bits - 1
    i_arr = np.array([i for i, _ in edges])
    j_arr = np.array([j for _, j in edges])
    bond_sum = (spins[:, i_arr] * spins[:, j_arr]).sum(axis=1)
    E_arr = -bond_sum.astype(np.float64) - h * spins.sum(axis=1)
    log_w = -beta * E_arr
    log_w -= log_w.max()
    w = np.exp(log_w)
    return float((E_arr * w).sum() / w.sum())


def nx_to_adj(G_nx: nx.Graph) -> Graph:
    return {v: list(G_nx.neighbors(v)) for v in G_nx.nodes()}


def _log_record_steps(n_steps: int, n_points: int) -> np.ndarray:
    return np.unique(np.round(np.geomspace(1, n_steps, num=n_points)).astype(int))


def run_chain_recording(
    G: Graph,
    beta: float, h: float, init: str, dynamics: str,
    n_steps: int, record_at_set: set, seed: int,
) -> Tuple[List[int], List[float]]:
    """Run the single-site dynamics for n_steps and return the running-mean
    energy sampled at the step counts in record_at_set.

    Energy is updated incrementally each step (cheaper than recomputing).
    """
    rng = random.Random(seed)
    chain = IsingChain(G, h=h, beta=beta, rng=rng, init=init, dynamics=dynamics)
    E = chain.energy()
    running_sum = 0.0
    steps_axis: List[int] = []
    running_mean: List[float] = []
    nodes = chain.nodes
    sigma = chain.sigma
    adj = chain.G
    e = math.e
    for t in range(1, n_steps + 1):
        v = rng.choice(nodes)
        neighbour_sum = sum(sigma[u] for u in adj[v])
        if dynamics == "metropolis":
            dE = 2.0 * sigma[v] * (neighbour_sum + h)
            if dE <= 0 or rng.random() < pow(e, -beta * dE):
                sigma[v] = -sigma[v]
                E += dE
        else:  # glauber
            z = 2.0 * beta * (neighbour_sum + h)
            if z >= 0:
                p_plus = 1.0 / (1.0 + pow(e, -z))
            else:
                ez = pow(e, z)
                p_plus = ez / (1.0 + ez)
            new = 1 if rng.random() < p_plus else -1
            if new != sigma[v]:
                dE = 2.0 * sigma[v] * (neighbour_sum + h)
                sigma[v] = new
                E += dE
        running_sum += E
        if t in record_at_set:
            steps_axis.append(t)
            running_mean.append(running_sum / t)
    return steps_axis, running_mean


def _existing_graph_ids(path: str) -> set:
    """Return the set of graph_id values already present in a CSV with a
    'graph_id' column.  Returns empty set if the file is missing."""
    out = set()
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.add(row["graph_id"])
    return out


def _existing_chains(path: str) -> set:
    """Set of (graph_id, h, beta, init, dynamics) chains already in
    traces.csv.  Lets us re-run only the chains we don't have yet."""
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.add((row["graph_id"], float(row["h"]),
                     float(row["beta"]), row["init"], row["dynamics"]))
    return out


def _existing_exact(path: str) -> set:
    """Set of (graph_id, h, beta) cells already in exact.csv."""
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.add((row["graph_id"], float(row["h"]), float(row["beta"])))
    return out


def main():
    os.makedirs(GRAPHS_SUBDIR, exist_ok=True)
    record_at = _log_record_steps(N_STEPS, NUM_LOG_SAMPLES)
    record_at_set = set(int(s) for s in record_at)

    done_chains = _existing_chains(TRACES_CSV)
    done_exact = _existing_exact(EXACT_CSV)
    print(f"existing chains in traces.csv: {len(done_chains)}, "
          f"exact cells: {len(done_exact)}")

    traces_exists = os.path.exists(TRACES_CSV)
    exact_exists = os.path.exists(EXACT_CSV)
    ft = open(TRACES_CSV, "a" if traces_exists else "w", newline="")
    fe = open(EXACT_CSV, "a" if exact_exists else "w", newline="")
    try:
        traces_writer = csv.writer(ft)
        exact_writer = csv.writer(fe)
        if not traces_exists:
            traces_writer.writerow(["graph_id", "n", "h", "beta", "init",
                                    "dynamics", "step", "running_mean_E"])
        if not exact_exists:
            exact_writer.writerow(["graph_id", "n", "h", "beta", "exact_E"])

        for n, base_seed, num_graphs in SIZES_SEEDS_COUNTS:
            for g_idx in range(num_graphs):
                graph_id = f"n{n}_graph{g_idx}"
                gpath = os.path.join(GRAPHS_SUBDIR, f"{graph_id}.json")
                if os.path.exists(gpath):
                    with open(gpath) as fj:
                        gj = json.load(fj)
                    edges = [tuple(e) for e in gj["edges"]]
                    G_nx = nx.Graph()
                    G_nx.add_nodes_from(range(n))
                    G_nx.add_edges_from(edges)
                else:
                    G_nx = nx.random_regular_graph(d=3, n=n,
                                                   seed=base_seed + g_idx)
                    with open(gpath, "w") as fj:
                        json.dump({"n": n, "g_idx": g_idx,
                                   "edges": [list(e) for e in G_nx.edges()]}, fj)
                    edges = list(G_nx.edges())
                G = nx_to_adj(G_nx)

                # Brute-force exact for small n: only cells we don't have yet.
                if n <= 20:
                    wrote_any_exact = False
                    for h in H_VALUES:
                        for beta in BETAS:
                            if (graph_id, h, beta) in done_exact:
                                continue
                            E_exact = exact_expected_energy(edges, n, beta, h=h)
                            exact_writer.writerow([graph_id, n, h, beta,
                                                   f"{E_exact:.10g}"])
                            wrote_any_exact = True
                    if wrote_any_exact:
                        fe.flush()

                # MCMC chains: skip ones we already have.
                for h_idx, h in enumerate(H_VALUES):
                    for init in INITS:
                        for dyn in DYNAMICS:
                            chains_to_run = [
                                (b_idx, beta) for b_idx, beta in enumerate(BETAS)
                                if (graph_id, h, beta, init, dyn) not in done_chains
                            ]
                            if not chains_to_run:
                                continue
                            print(f"  {graph_id}  h={h}  init={init}  dyn={dyn}  "
                                  f"({len(chains_to_run)} betas)")
                            for b_idx, beta in chains_to_run:
                                seed = (1000 * (base_seed + g_idx)
                                        + 100 * h_idx
                                        + (0 if init == "ground" else 50_000)
                                        + (0 if dyn == "metropolis" else 1_000_000)
                                        + b_idx)
                                steps, mean_E = run_chain_recording(
                                    G, beta, h, init, dyn,
                                    N_STEPS, record_at_set, seed,
                                )
                                for s, m in zip(steps, mean_E):
                                    traces_writer.writerow(
                                        [graph_id, n, h, beta, init, dyn, s, f"{m:.6g}"]
                                    )
                            ft.flush()
    finally:
        ft.close()
        fe.close()

    print(f"\nwrote {TRACES_CSV}")
    print(f"wrote {EXACT_CSV}")
    print(f"wrote graph adjacency lists under {GRAPHS_SUBDIR}/")


if __name__ == "__main__":
    main()
