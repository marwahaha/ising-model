"""High-temperature expansion (HTE) log Z estimate at h = 0.

For a 3-regular graph (|E| = 3|V|/2) at h = 0,

    log Z(beta) ~ |V| * log 2 + |E| * log cosh(beta)
                 + sum_{k=3}^{K} f(k) * log(1 + tanh(beta)^k)

Two variants for f(k):

  variant="asym"
    f(k) = 2^k / (2k) for all k >= 3 -- the asymptotic expected number of
    k-cycles in a random 3-regular graph.  Graph-independent.

  variant="exact"
    f(k) = true #k-cycles in this graph for k = 3, 4, 5 (counted by direct
    enumeration: O(n * d^{k-1}) which is trivial for our sizes), and
    asymptotic for k >= 6 (enumeration cost grows; gain is small once the
    series enters the divergent regime).  Graph-specific.

Output: data/log_z_hte.csv
  columns: graph_id, n, h, beta, K, variant, log_Z
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List, Tuple


BETAS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 5.0]
H_VALUES = [0.0]
VARIANTS = ["asym", "exact"]
GRAPHS_TO_RUN = {"n16_graph0", "n16_graph1", "n16_graph2", "n16_graph3",
                 "n30_graph0", "n30_graph1", "n40_graph0", "n40_graph1",
                 "n50_graph0", "n50_graph1",
                 "n60_graph0", "n60_graph1"}

DATA_DIR = "data"
GRAPHS_SUBDIR = os.path.join(DATA_DIR, "graphs")
OUT_CSV = os.path.join(DATA_DIR, "log_z_hte.csv")


def _asym(k: int) -> float:
    return (2.0 ** k) / (2.0 * k)


def adj_from_edges(n: int, edges) -> List[List[int]]:
    G: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        G[u].append(v)
        G[v].append(u)
    return G


def count_3_cycles(adj: List[List[int]], n: int) -> int:
    c = 0
    for u in range(n):
        for a in adj[u]:
            if a <= u:
                continue
            for b in adj[a]:
                if b <= u or b == a:
                    continue
                if u in adj[b]:
                    c += 1
    return c // 2


def count_4_cycles(adj: List[List[int]], n: int) -> int:
    c = 0
    for u in range(n):
        for a in adj[u]:
            if a <= u:
                continue
            for b in adj[a]:
                if b == u or b <= u:
                    continue
                for d in adj[b]:
                    if d == u or d == a or d <= u:
                        continue
                    if u in adj[d]:
                        c += 1
    return c // 2


def count_5_cycles(adj: List[List[int]], n: int) -> int:
    c = 0
    for u in range(n):
        for a in adj[u]:
            if a <= u:
                continue
            for b in adj[a]:
                if b == u or b <= u:
                    continue
                for cc in adj[b]:
                    if cc in (u, a) or cc <= u:
                        continue
                    for d in adj[cc]:
                        if d in (u, a, b) or d <= u:
                            continue
                        if u in adj[d]:
                            c += 1
    return c // 2


def exact_cycle_counts(adj: List[List[int]], n: int) -> Dict[int, int]:
    """Exact #k-cycles for k in {3, 4, 5} via direct enumeration."""
    return {3: count_3_cycles(adj, n),
            4: count_4_cycles(adj, n),
            5: count_5_cycles(adj, n)}


def hte_log_z(n: int, m: int, beta: float, K: int,
              f_override: Dict[int, float] | None = None) -> float:
    base = n * math.log(2.0) + m * math.log(math.cosh(beta))
    t = math.tanh(beta)
    if abs(t) < 1e-15 or K < 3:
        return base
    s = 0.0
    for k in range(3, K + 1):
        if f_override and k in f_override:
            f = f_override[k]
        else:
            f = _asym(k)
        s += f * math.log1p(t ** k)
    return base + s


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

    rows = 0
    with open(OUT_CSV, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["graph_id", "n", "h", "beta", "K", "variant", "log_Z"])
        for graph_id in graph_ids:
            with open(os.path.join(GRAPHS_SUBDIR, f"{graph_id}.json")) as f:
                gj = json.load(f)
            n = gj["n"]
            edges = [tuple(e) for e in gj["edges"]]
            m = len(edges)
            adj = adj_from_edges(n, edges)
            exact = exact_cycle_counts(adj, n)
            f_exact = {k: float(v) for k, v in exact.items()}
            print(f"{graph_id}: cycles (3,4,5) = "
                  f"({exact[3]}, {exact[4]}, {exact[5]})")
            for h in H_VALUES:
                for beta in BETAS:
                    for K in range(3, n + 1):
                        lz_asym = hte_log_z(n, m, beta, K, f_override=None)
                        lz_exact = hte_log_z(n, m, beta, K, f_override=f_exact)
                        w.writerow([graph_id, n, f"{h:.1f}", f"{beta:.6g}",
                                    K, "asym", f"{lz_asym:.6f}"])
                        w.writerow([graph_id, n, f"{h:.1f}", f"{beta:.6g}",
                                    K, "exact", f"{lz_exact:.6f}"])
                        rows += 2
    print(f"Wrote {rows} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
