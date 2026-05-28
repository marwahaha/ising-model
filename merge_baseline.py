"""Merge per-graph baseline partials into data/log_z_fep_baseline.csv and
report validation stats vs exact log Z (where available).

Reads:  data/baseline_partials/*.csv
Writes: data/log_z_fep_baseline.csv (concatenated, one CSV header)

Also prints relative-error summary by (n, h) vs data/log_z.csv exact rows.
"""

from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict


PARTIALS_DIR = "data/baseline_partials"
OUT_CSV = "data/log_z_fep_baseline.csv"
EXACT_CSV = "data/log_z.csv"
ANCHORS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 5.0]


def merge() -> int:
    rows = 0
    header_written = False
    with open(OUT_CSV, "w", newline="") as fout:
        w = csv.writer(fout)
        for path in sorted(glob.glob(os.path.join(PARTIALS_DIR, "*.csv"))):
            with open(path, newline="") as fin:
                r = csv.reader(fin)
                header = next(r, None)
                if header is None:
                    continue
                if not header_written:
                    w.writerow(header)
                    header_written = True
                for row in r:
                    w.writerow(row)
                    rows += 1
    return rows


def load_exact() -> dict:
    out = {}
    if not os.path.exists(EXACT_CSV):
        return out
    with open(EXACT_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] != "exact":
                continue
            out[(r["graph_id"], float(r["h"]), float(r["beta"]))] = float(r["log_Z"])
    return out


def load_baseline_final() -> dict:
    """Return {(graph,h,init,dyn,beta): final-step log_Z}."""
    last = defaultdict(lambda: (-1, None))
    with open(OUT_CSV, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["graph_id"], float(r["h"]),
                   r["init"], r["dynamics"], float(r["beta"]))
            step = int(r["step"])
            if step > last[key][0]:
                last[key] = (step, float(r["log_Z"]))
    return {k: v[1] for k, v in last.items()}


def main() -> None:
    n_rows = merge()
    print(f"Merged {n_rows} rows -> {OUT_CSV}")
    exact = load_exact()
    bl = load_baseline_final()
    by_n = defaultdict(list)
    by_n_h = defaultdict(list)
    for (g, h, init, dyn, b), v in bl.items():
        ex = exact.get((g, h, b))
        if ex is None:
            continue
        re = abs(v - ex) / abs(ex)
        n = int(g.split("_")[0][1:])
        by_n[n].append(re)
        by_n_h[(n, h)].append(re)

    print("\nValidation (vs exact log Z, anchor betas):")
    print(f"{'n':>4}{'pts':>6}{'mean rel err':>16}{'max':>12}")
    for n in sorted(by_n):
        a = by_n[n]
        print(f"{n:>4}{len(a):>6}{sum(a)/len(a):>16.3e}{max(a):>12.3e}")

    if by_n_h:
        print("\nPer (n, h):")
        print(f"{'n':>4}{'h':>6}{'pts':>5}{'mean':>14}{'max':>12}")
        for (n, h) in sorted(by_n_h):
            a = by_n_h[(n, h)]
            print(f"{n:>4}{h:>6.1f}{len(a):>5}{sum(a)/len(a):>14.3e}{max(a):>12.3e}")

    graphs_in_baseline = sorted({k[0] for k in bl})
    print(f"\nGraphs in baseline ({len(graphs_in_baseline)}): "
          f"{graphs_in_baseline}")


if __name__ == "__main__":
    main()
