"""High-temperature expansion (HTE) log Z estimate.  Only h = 0.

For a 3-regular graph (|E| = 3|V|/2) at h = 0,

    log Z(beta) ~ |V| * log 2 + |E| * log cosh(beta)
                 + sum_{k=3}^{K} f(k) * log(1 + tanh(beta)^k)

with f(k) = 2^k / (2k), the asymptotic expected number of k-cycles in a
random 3-regular graph.  The estimate is graph-independent at fixed n;
to fit the plot.py CSV schema we still write one row per graph.

The series f(k)*log(1+tanh(beta)^k) diverges for tanh(beta) > 1/2 ~
beta > 0.55 once k is large enough, since f(k) ~ 2^k.  We therefore
report two short truncations per n:
    K_short = 3                  (just the triangle correction)
    K_log   = round(2 * ln(n))   (~log-n scaling)
Beyond ~log n the asymptotic cycle count overcounts the actual cycles
in a finite graph and the partial sum is dominated by spurious tail.

Output: data/log_z_hte.csv
  columns: graph_id, n, h, beta, K, log_Z
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import List, Tuple


BETAS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 5.0]
H_VALUES = [0.0]
GRAPHS_TO_RUN = {"n16_graph0", "n16_graph1", "n16_graph2", "n16_graph3",
                 "n30_graph0", "n30_graph1", "n40_graph0", "n40_graph1",
                 "n50_graph0", "n50_graph1",
                 "n60_graph0", "n60_graph1"}

DATA_DIR = "data"
GRAPHS_SUBDIR = os.path.join(DATA_DIR, "graphs")
OUT_CSV = os.path.join(DATA_DIR, "log_z_hte.csv")


def _f(k: int) -> float:
    return (2.0 ** k) / (2.0 * k)


def hte_log_z(n: int, m: int, beta: float, K: int) -> float:
    """HTE truncation up to k=K (inclusive).  Cycles of length 1, 2 do not
    occur on simple graphs, so the sum starts at k=3."""
    base = n * math.log(2.0) + m * math.log(math.cosh(beta))
    t = math.tanh(beta)
    if abs(t) < 1e-15 or K < 3:
        return base
    s = 0.0
    for k in range(3, K + 1):
        s += _f(k) * math.log1p(t ** k)
    return base + s


def _k_values(n: int) -> List[int]:
    """Every integer truncation from k=3 to k=n.  Past k=~log n the
    asymptotic f(k)=2^k/(2k) overcounts cycles in a finite graph and the
    partial sum diverges -- shown deliberately so the user sees the
    truncation trade-off."""
    return list(range(3, n + 1))


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
        w.writerow(["graph_id", "n", "h", "beta", "K", "log_Z"])
        for graph_id in graph_ids:
            with open(os.path.join(GRAPHS_SUBDIR, f"{graph_id}.json")) as f:
                gj = json.load(f)
            n = gj["n"]
            m = len(gj["edges"])
            for h in H_VALUES:
                for beta in BETAS:
                    for K in _k_values(n):
                        lz = hte_log_z(n, m, beta, K)
                        w.writerow([graph_id, n, f"{h:.1f}", f"{beta:.6g}",
                                    K, f"{lz:.6f}"])
                        rows += 1
    print(f"Wrote {rows} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
