"""Bidirectional dense-ladder FEP (Glauber).

For each beta in the ladder, run ONE chain (post-burnin samples).  Use it
for two estimators:
  - Forward leg of segment (b_k -> b_{k+1}):   log < e^{-Db * E} >_{b_k}
  - Reverse leg of segment (b_{k-1} -> b_k):  -log < e^{+Db * E} >_{b_k}

Each adjacent pair (b_k, b_{k+1}) then yields two estimates of the same
ratio log Z(b_{k+1}) - log Z(b_k):
    forward_k:  +log < e^{-Db * E} >_{b_k}
    reverse_k:  -log < e^{+Db * E} >_{b_{k+1}}
We use the simple symmetric average (forward + reverse) / 2 -- this
cancels the leading O(Db^2 * Var(E)) bias from finite-sample log-of-mean
(by Jensen) and is what BAR reduces to in the small-Db limit.

Output: data/log_z_fep_bidir.csv  (same schema as run_fep.py).
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
    "FEP_OUT_CSV", os.path.join(DATA_DIR, "log_z_fep_bidir.csv"))


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


def chain_bidir(G, n, edges, beta, h, dbeta_fwd, dbeta_rev,
                init, dynamics, record_set, seed):
    """Run one chain at `beta`.  Accumulate two online log-mean-exp values:
      - lmexp_fwd[k] = log < e^{-dbeta_fwd * E} >_{beta} at sample k  (forward
        ratio toward beta + dbeta_fwd)
      - lmexp_rev[k] = log < e^{+dbeta_rev * E} >_{beta} at sample k  (reverse
        leg of the segment beta - dbeta_rev -> beta)
    dbeta_fwd/rev are 0 to skip the corresponding endpoint (k=0 has no rev,
    k=K has no fwd)."""
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

    do_fwd = dbeta_fwd > 0.0
    do_rev = dbeta_rev > 0.0
    m_f = -math.inf; S_f = 0.0
    m_r = -math.inf; S_r = 0.0
    cnt = 0
    out_fwd: Dict[int, float] = {}
    out_rev: Dict[int, float] = {}
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
            cnt += 1
            if do_fwd:
                x = -dbeta_fwd * E
                if cnt == 1 or x > m_f:
                    if cnt > 1:
                        S_f = S_f * math.exp(m_f - x) + 1.0
                    else:
                        S_f = 1.0
                    m_f = x
                else:
                    S_f += math.exp(x - m_f)
            if do_rev:
                x = +dbeta_rev * E
                if cnt == 1 or x > m_r:
                    if cnt > 1:
                        S_r = S_r * math.exp(m_r - x) + 1.0
                    else:
                        S_r = 1.0
                    m_r = x
                else:
                    S_r += math.exp(x - m_r)
            if cnt in record_set:
                if do_fwd:
                    out_fwd[cnt] = m_f + math.log(S_f) - math.log(cnt)
                if do_rev:
                    out_rev[cnt] = m_r + math.log(S_r) - math.log(cnt)
    return out_fwd, out_rev


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
    print(f"Per-chain: burnin={BURNIN}, collect={COLLECT}")
    print(f"Graphs: {graph_ids}  inits={INITS}  dyns={DYNS}  h={H_VALUES}")

    K = len(LADDER) - 1  # number of segments

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
                        # Per ladder index i in [0..K]: chain at beta = LADDER[i]
                        #   fwd estimator uses dbeta = LADDER[i+1]-LADDER[i]  (i < K)
                        #   rev estimator uses dbeta = LADDER[i]-LADDER[i-1]  (i > 0)
                        fwd_seg: List[Dict[int, float]] = []  # i in [0..K-1]
                        rev_seg: List[Dict[int, float]] = []  # i in [1..K]
                        for i in range(K + 1):
                            beta_i = LADDER[i]
                            dbf = (LADDER[i + 1] - LADDER[i]) if i < K else 0.0
                            dbr = (LADDER[i] - LADDER[i - 1]) if i > 0 else 0.0
                            seed = abs(hash((graph_id, h, beta_i, init, dyn,
                                             "fep_bidir"))) % (1 << 32)
                            ofwd, orev = chain_bidir(
                                G, n, edges, beta_i, h, dbf, dbr,
                                init, dyn, record_set, seed)
                            if i < K:
                                fwd_seg.append(ofwd)
                            if i > 0:
                                rev_seg.append(orev)
                        # rev_seg[i-1] is reverse-leg estimate for segment i
                        base = n * math.log(2.0)
                        for k in range(K):
                            beta = LADDER[k + 1]
                            for step in record_pts:
                                # segment j (j in 0..k): avg forward + reverse
                                lz = base
                                for j in range(k + 1):
                                    fwd = fwd_seg[j][step]
                                    rev = -rev_seg[j][step]  # reverse leg
                                    lz += 0.5 * (fwd + rev)
                                w.writerow([graph_id, n, h, init, dyn,
                                            beta, step, f"{lz:.6f}"])
                        fout.flush()
                        print(f"  h={h:.1f} init={init} dyn={dyn} done",
                              flush=True)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
