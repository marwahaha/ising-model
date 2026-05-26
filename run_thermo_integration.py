"""Derive log Z(beta, h) from the same Metropolis/Glauber chains used to
estimate <E>.  No new MCMC -- this is a pure post-processing pass over
data/traces.csv produced by simulate.py.

Why this works: from the same chain we can read log Z by integrating <E>
in beta, because

    d/dbeta log Z(beta, h) = -<E>(beta, h)
    log Z(0, h)            = n * log 2     (uniform measure over 2^n configs)

so

    log Z(beta, h)  =  n * log 2  -  integral_0^beta <E>(beta', h) dbeta'.

simulate.py records running-mean <E>(beta, h, init, dyn, step) at every
log-spaced step.  At each fixed step we have <E> at every beta in the
BETAS grid; trapezoidal-integrate in beta (prepending the exact endpoint
(beta=0, <E>=0)) to get log Z(beta_i, h, init, dyn, step) at every recorded
step.  The resulting trajectory is the same chains' running estimate of
log Z under thermo integration.

Inputs : data/traces.csv  (from simulate.py)
Outputs: data/log_z_mcmc.csv
           columns: graph_id, n, h, beta, init, dynamics, step, log_z_mcmc
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict

DATA_DIR = "data"
TRACES_CSV = os.path.join(DATA_DIR, "traces.csv")
OUT_CSV = os.path.join(DATA_DIR, "log_z_mcmc.csv")


def main():
    # by_key[(gid, h, init, dyn, step)] = { beta: <E> }
    by_key: dict = defaultdict(dict)
    n_per_graph: dict = {}
    with open(TRACES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            gid = row["graph_id"]
            n_per_graph[gid] = int(row["n"])
            key = (gid, float(row["h"]), row["init"], row["dynamics"],
                   int(row["step"]))
            by_key[key][float(row["beta"])] = float(row["running_mean_E"])

    print(f"loaded {len(by_key):,} (graph,h,init,dyn,step) tuples "
          f"across {len(n_per_graph)} graphs")

    with open(OUT_CSV, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["graph_id", "n", "h", "beta", "init", "dynamics",
                    "step", "log_z_mcmc"])
        for key, beta_to_E in by_key.items():
            gid, h, init, dyn, step = key
            n = n_per_graph[gid]
            pts = [(0.0, 0.0)] + sorted(beta_to_E.items())
            log_Z = n * math.log(2.0)
            for i in range(1, len(pts)):
                b_prev, e_prev = pts[i - 1]
                b_curr, e_curr = pts[i]
                log_Z -= 0.5 * (b_curr - b_prev) * (e_curr + e_prev)
                w.writerow([gid, n, h, b_curr, init, dyn, step,
                            f"{log_Z:.6f}"])

    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
