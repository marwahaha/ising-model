"""Microbench per-step wall cost of each estimator, per graph size.

The wall-time view (plot.py) converts step counts to seconds using a
per-step rate.  That rate drifts with n (cache effects, neighbour-sum
loops), so we measure one number per size and per kind.  Run this with
NOTHING ELSE computing on the machine, then paste the printed dict into
plot.py's US_PER_STEP_BY_N.

Usage:  python3 run_microbench.py
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, List, Tuple

from ising import IsingChain
from subgraphs import SubgraphsChain

GRAPHS_SUBDIR = os.path.join("data", "graphs")
# One representative graph per size.
SIZE_GRAPHS = {16: "n16_graph0", 30: "n30_graph0", 40: "n40_graph0",
               50: "n50_graph0", 60: "n60_graph0"}
BETA = 0.5
WARMUP = 20_000
MEASURE = 400_000
REPEATS = 3


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


def _best_us(make_chain, warmup: int, measure: int, repeats: int) -> float:
    """Return the fastest (min over repeats) µs/step -- min is the most
    stable estimator of intrinsic cost, least polluted by scheduler noise."""
    best = math.inf
    for _ in range(repeats):
        chain = make_chain()
        for _ in range(warmup):
            chain.step()
        t0 = time.perf_counter()
        for _ in range(measure):
            chain.step()
        us = (time.perf_counter() - t0) / measure * 1e6
        best = min(best, us)
    return best


def main() -> None:
    out: Dict[int, Dict[str, float]] = {}
    print(f"{'n':>4} {'metropolis':>12} {'glauber':>12} {'js':>12}")
    for n in sorted(SIZE_GRAPHS):
        edges, nn = load_graph(SIZE_GRAPHS[n])
        assert nn == n
        G = nx_adj(n, edges)
        rates: Dict[str, float] = {}
        for dyn in ("metropolis", "glauber"):
            rates[dyn] = _best_us(
                lambda dyn=dyn: IsingChain(G, h=0.0, beta=BETA,
                                           rng=random.Random(0),
                                           init="uniform", dynamics=dyn),
                WARMUP, MEASURE, REPEATS)
        edge_lambdas = [math.tanh(BETA)] * len(edges)
        rates["js"] = _best_us(
            lambda: SubgraphsChain(edges, n, edge_lambdas=edge_lambdas,
                                   mu=0.0, rng=random.Random(0)),
            WARMUP, MEASURE, REPEATS)
        out[n] = {k: round(v, 3) for k, v in rates.items()}
        print(f"{n:>4} {rates['metropolis']:>12.3f} "
              f"{rates['glauber']:>12.3f} {rates['js']:>12.3f}")

    print("\nUS_PER_STEP_BY_N = {")
    for n in sorted(out):
        r = out[n]
        print(f"    {n}: {{\"metropolis\": {r['metropolis']}, "
              f"\"glauber\": {r['glauber']}, \"js\": {r['js']}}},")
    print("}")


if __name__ == "__main__":
    main()
