"""Microbench FEP per-step wall cost, per graph size and dynamics.

The wall-time view converts FEP chain-step counts to seconds via a per-step
rate.  A FEP step = one single-site Glauber/Metropolis update plus an online
log-sum-exp accumulation, so it is a bit dearer than a bare spin flip; we
measure run_fep's actual inner loop (segment_running_logmeanexp) at the
production burn-in/collect proportions.  Run with NOTHING else computing,
then paste the printed dict into plot.py's FEP_US_PER_STEP_BY_N.
"""

import json
import math
import os
import time

import run_fep

GRAPHS_SUBDIR = os.path.join("data", "graphs")
SIZE_GRAPHS = {16: "n16_graph0", 30: "n30_graph0",
               40: "n40_graph0", 50: "n50_graph0"}
BETA, H, DBETA = 0.5, 0.2, 0.1
BURNIN, COLLECT = 3_000, 150_000       # match the producer's proportions
REPEATS = 3


def load(gid):
    with open(os.path.join(GRAPHS_SUBDIR, f"{gid}.json")) as f:
        g = json.load(f)
    return [tuple(e) for e in g["edges"]], g["n"]


def best_us(G, n, edges, dyn):
    run_fep.BURNIN, run_fep.COLLECT = BURNIN, COLLECT
    best = math.inf
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        run_fep.segment_running_logmeanexp(
            G, n, edges, BETA, H, DBETA, "uniform", dyn, set(), 0)
        el = time.perf_counter() - t0
        best = min(best, el / (BURNIN + COLLECT) * 1e6)
    return best


def main():
    out = {}
    print(f"{'n':>4} {'fep_metropolis':>15} {'fep_glauber':>13}")
    for n in sorted(SIZE_GRAPHS):
        edges, nn = load(SIZE_GRAPHS[n])
        assert nn == n
        G = run_fep.nx_adj(n, edges)
        m = best_us(G, n, edges, "metropolis")
        g = best_us(G, n, edges, "glauber")
        out[n] = {"metropolis": round(m, 3), "glauber": round(g, 3)}
        print(f"{n:>4} {m:>15.3f} {g:>13.3f}", flush=True)
    print("\nFEP_US_PER_STEP_BY_N = {")
    for n in sorted(out):
        r = out[n]
        print(f'    {n}: {{"metropolis": {r["metropolis"]}, '
              f'"glauber": {r["glauber"]}}},')
    print("}")


if __name__ == "__main__":
    main()
