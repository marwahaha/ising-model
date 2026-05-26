"""Quick comparison: LSS Taylor estimate vs exact log Z, on n16_graph0.

Uses the 'naive' backend (enumerate subsets of size <= m), which on n=16 is
roughly C(16, m) per coefficient -- fast for m up to ~10."""

import csv
import json
import sys
import time
sys.path.insert(0, ".")
from adapter import estimate_log_Z, log_z_error_bound


def load_graph(path):
    g = json.load(open(path))
    return [tuple(e) for e in g["edges"]], g["n"]


def main():
    edges, n = load_graph("../data/graphs/n16_graph0.json")
    print(f"n={n}, |E|={len(edges)}")

    exact = {}
    for row in csv.DictReader(open("../data/log_z.csv")):
        if row["graph_id"] == "n16_graph0" and row["method"] == "exact":
            exact[(float(row["h"]), float(row["beta"]))] = float(row["log_Z"])

    print(f"{'h':>4} {'beta':>5} {'m':>3} "
          f"{'exact':>10} {'lss':>10} {'err':>10} {'bound':>10}   t(s)")
    cases = [(0.1, 0.5), (0.1, 1.0), (0.5, 0.5), (0.5, 1.0), (1.0, 0.5)]
    for h, beta in cases:
        lz_exact = exact[(h, beta)]
        for m in (2, 4, 6, 8, 10):
            t0 = time.time()
            lz_lss = estimate_log_Z(edges, n, beta, h, m, method="naive")
            dt = time.time() - t0
            err = abs(lz_lss - lz_exact)
            bnd = log_z_error_bound(n, beta, h, m)
            print(f"{h:>4.1f} {beta:>5.2f} {m:>3} "
                  f"{lz_exact:>10.5f} {lz_lss:>10.5f} "
                  f"{err:>10.2e} {bnd:>10.2e}   {dt:.2f}")


if __name__ == "__main__":
    main()
