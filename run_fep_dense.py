"""Dense-ladder Free-energy-perturbation telescoping estimate (Glauber only).

A higher-accuracy variant of run_fep.py that trades runtime for a denser
beta ladder and longer chains per segment.  Forward-FEP recursion is

    log Z(b_{k+1}) = log Z(b_k) + log < e^{-Db_k * E} >_{b_k},

starting from log Z(0) = n*log2.  Variance of each ratio shrinks with
Db_k, so a denser ladder + longer chains yields a tighter telescoped
estimate at every anchor beta.

This script defaults to Glauber dynamics and to the n=16 graphs (where
exact log Z is available for validation in data/log_z.csv).  See
GRAPHS_TO_RUN to extend.

Output: data/log_z_fep_dense.csv
  columns: graph_id, n, h, init, dynamics, beta, step, log_Z
Incremental: skips (graph_id, h, init, dynamics) already present.

Beta grid is uniform Db = 0.05 from 0 to 5.0 (100 segments).  Log Z is
recorded at every beta in the ladder, on a log-spaced step grid, so
convergence-vs-work plots are still available; final-step rows give the
best estimate.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np

def _env_list(name: str, default: List) -> List:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return [type(default[0])(x) for x in raw.split(",") if x.strip()]


DBETA = float(os.environ.get("FEP_DBETA", "0.05"))
BETA_MAX = float(os.environ.get("FEP_BETA_MAX", "5.0"))
LADDER = [round(i * DBETA, 6) for i in range(int(BETA_MAX / DBETA) + 1)]
H_VALUES = _env_list("FEP_H_VALUES", [0.0, 0.1, 0.2, 0.5, 1.0])
INITS = _env_list("FEP_INITS", ["ground", "uniform"])
DYNS = _env_list("FEP_DYNS", ["glauber"])
GRAPHS_TO_RUN = set(_env_list(
    "FEP_GRAPHS",
    ["n16_graph0", "n16_graph1", "n16_graph2", "n16_graph3"]))

BURNIN = int(os.environ.get("FEP_BURNIN", "5000"))
COLLECT = int(os.environ.get("FEP_COLLECT", "500000"))
NUM_LOG_POINTS = 30

DATA_DIR = "data"
GRAPHS_SUBDIR = os.path.join(DATA_DIR, "graphs")
OUT_CSV = os.environ.get(
    "FEP_OUT_CSV", os.path.join(DATA_DIR, "log_z_fep_dense.csv"))


def load_graph(graph_id: str) -> Tuple[List[Tuple[int, int]], int]:
    with open(os.path.join(GRAPHS_SUBDIR, f"{graph_id}.json")) as f:
        g = json.load(f)
    return [tuple(e) for e in g["edges"]], g["n"]


def nx_adj(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    G: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        G[u].append(v)
        G[v].append(u)
    return G


def _record_points() -> List[int]:
    pts = np.unique(np.geomspace(1, COLLECT, NUM_LOG_POINTS).astype(np.int64))
    pts = pts[(pts >= 1) & (pts <= COLLECT)]
    if pts[-1] != COLLECT:
        pts = np.append(pts, COLLECT)
    return [int(x) for x in pts]


def segment_running_logmeanexp(G, n, edges, beta, h, dbeta, init, dynamics,
                               record_set, seed) -> Dict[int, float]:
    """Run a Glauber/Metropolis chain at `beta` under (init, dynamics); return
    {collect_index: running log<e^{-dbeta*E}>} at the requested post-burn-in
    indices, via an online log-sum-exp accumulator (O(1) per sample)."""
    rng = random.Random(seed)
    if init == "ground":
        if h > 0:
            s0 = 1
        elif h < 0:
            s0 = -1
        else:
            s0 = 1 if rng.random() < 0.5 else -1
        sigma = [s0] * n
    else:
        sigma = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
    bond = sum(sigma[u] * sigma[v] for u, v in edges)
    M = sum(sigma)
    E = -bond - h * M
    e = math.e
    glauber = (dynamics == "glauber")
    m = -math.inf
    S = 0.0
    cnt = 0
    out: Dict[int, float] = {}
    total = BURNIN + COLLECT
    for t in range(1, total + 1):
        v = rng.randrange(n)
        nb = 0
        for u in G[v]:
            nb += sigma[u]
        dE = 2.0 * sigma[v] * (nb + h)
        if glauber:
            z = 2.0 * beta * (nb + h)
            if z >= 0:
                p = 1.0 / (1.0 + pow(e, -z))
            else:
                ez = pow(e, z)
                p = ez / (1.0 + ez)
            new = 1 if rng.random() < p else -1
            if new != sigma[v]:
                E += dE
                sigma[v] = new
        else:
            if dE <= 0.0 or rng.random() < pow(e, -beta * dE):
                E += dE
                sigma[v] = -sigma[v]
        if t > BURNIN:
            x = -dbeta * E
            cnt += 1
            if x > m:
                S = (S * math.exp(m - x) + 1.0) if cnt > 1 else 1.0
                m = x
            else:
                S += math.exp(x - m)
            if cnt in record_set:
                out[cnt] = m + math.log(S) - math.log(cnt)
    return out


def _existing(path: str) -> set:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.add((row["graph_id"], float(row["h"]),
                     row["init"], row["dynamics"]))
    return out


def _sort_key(gid: str) -> Tuple[int, int]:
    n_part, g_part = gid.split("_")
    return int(n_part[1:]), int(g_part.replace("graph", ""))


def main() -> None:
    graph_ids = sorted(
        (fn[:-len(".json")] for fn in os.listdir(GRAPHS_SUBDIR)
         if fn.endswith(".json")),
        key=_sort_key,
    )
    graph_ids = [g for g in graph_ids if g in GRAPHS_TO_RUN]
    record_pts = _record_points()
    record_set = set(record_pts)
    done = _existing(OUT_CSV)
    write_header = not os.path.exists(OUT_CSV)
    mode = "w" if write_header else "a"

    print(f"Ladder: {len(LADDER)} points, Db={DBETA}, beta_max={BETA_MAX}")
    print(f"Per-segment chain: burnin={BURNIN}, collect={COLLECT}")
    print(f"Graphs: {graph_ids}")

    with open(OUT_CSV, mode, newline="") as fout:
        w = csv.writer(fout)
        if write_header:
            w.writerow(["graph_id", "n", "h", "init", "dynamics",
                        "beta", "step", "log_Z"])
        for graph_id in graph_ids:
            edges, n = load_graph(graph_id)
            G = nx_adj(n, edges)
            print(f"\n=== {graph_id} (n={n}, |E|={len(edges)}) ===", flush=True)
            for h in H_VALUES:
                for init in INITS:
                    for dyn in DYNS:
                        if (graph_id, h, init, dyn) in done:
                            continue
                        seg: List[Dict[int, float]] = []
                        for k in range(len(LADDER) - 1):
                            bk = LADDER[k]
                            dbk = LADDER[k + 1] - LADDER[k]
                            seed = abs(hash((graph_id, h, bk, init, dyn,
                                             "fep_dense"))) % (1 << 32)
                            seg.append(segment_running_logmeanexp(
                                G, n, edges, bk, h, dbk, init, dyn,
                                record_set, seed))
                        base = n * math.log(2.0)
                        for k in range(len(LADDER) - 1):
                            beta = LADDER[k + 1]
                            for step in record_pts:
                                lz = base + sum(seg[j][step]
                                                for j in range(k + 1))
                                w.writerow([graph_id, n, h, init, dyn,
                                            beta, step, f"{lz:.6f}"])
                        fout.flush()
                        print(f"  h={h:.1f} init={init} dyn={dyn} done",
                              flush=True)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
