"""Compare dense FEP estimates to exact log Z (and coarse FEP) at anchors.

Reads:
  - data/log_z.csv (exact log Z, method='exact')
  - data/log_z_fep.csv (coarse-ladder FEP, full chain)
  - data/log_z_fep_dense.csv (dense-ladder FEP, full chain) -- override via
    env FEP_DENSE_CSV.

Reports relative-error |log_Z - exact| / |exact| at each anchor beta, for
both estimators and for every (graph_id, h, init) pair found in the dense
CSV.  Coarse FEP shown only when present in data/log_z_fep.csv.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict


ANCHORS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 5.0]
DENSE_CSV = os.environ.get("FEP_DENSE_CSV", "data/log_z_fep_dense.csv")


def load_exact() -> dict:
    out = {}
    with open("data/log_z.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] != "exact":
                continue
            out[(r["graph_id"], float(r["h"]), float(r["beta"]))] = float(r["log_Z"])
    return out


def load_fep(path: str) -> dict:
    """Return {(graph,h,init,dyn,beta): final_step_log_Z}."""
    if not os.path.exists(path):
        return {}
    last_step = defaultdict(lambda: (-1, None))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["graph_id"], float(r["h"]),
                   r["init"], r["dynamics"], float(r["beta"]))
            step = int(r["step"])
            cur, _ = last_step[key]
            if step > cur:
                last_step[key] = (step, float(r["log_Z"]))
    return {k: v[1] for k, v in last_step.items()}


def fmt(x):
    return "  --  " if x is None else f"{x:7.2e}"


def main() -> None:
    exact = load_exact()
    coarse = load_fep("data/log_z_fep.csv")
    dense = load_fep(DENSE_CSV)

    keys = sorted({(g, h, init, dyn) for (g, h, init, dyn, _) in dense})
    print(f"Dense CSV: {DENSE_CSV}")
    print(f"{'graph':<14}{'h':>5}{'init':>9}{'dyn':>10}{'beta':>7}"
          f"{'|err|/|exact| dense':>22}{'coarse':>12}{'exact':>11}"
          f"{'dense':>11}")
    print("-" * 110)

    totals_dense, totals_coarse, n_pts = 0.0, 0.0, 0
    n_coarse = 0
    for (g, h, init, dyn) in keys:
        for b in ANCHORS:
            ex = exact.get((g, h, b))
            if ex is None:
                continue
            d = dense.get((g, h, init, dyn, b))
            c = coarse.get((g, h, init, dyn, b))
            if d is None:
                continue
            re_d = abs(d - ex) / abs(ex)
            re_c = abs(c - ex) / abs(ex) if c is not None else None
            totals_dense += re_d
            n_pts += 1
            if re_c is not None:
                totals_coarse += re_c
                n_coarse += 1
            print(f"{g:<14}{h:>5.1f}{init:>9}{dyn:>10}{b:>7.2f}"
                  f"{fmt(re_d):>22}{fmt(re_c):>12}{ex:>11.4f}{d:>11.4f}")

    if n_pts:
        print("-" * 110)
        print(f"mean rel-err  dense: {totals_dense / n_pts:.3e}"
              f"   ({n_pts} pts)")
        if n_coarse:
            print(f"mean rel-err coarse: {totals_coarse / n_coarse:.3e}"
                  f"   ({n_coarse} pts)")


if __name__ == "__main__":
    main()
