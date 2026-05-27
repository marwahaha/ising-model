"""Produce Barvinok / Patel-Regts Taylor log Z estimates on every graph
under data/graphs/, across the project's (beta, h) grid, for both
backends (naive and insects) at a range of truncation orders m.

For h = 0 rows the Taylor estimator runs at h_eff = 1/n (Jerrum-Sinclair
field-anneal trick, to get off the Lee-Yang circle).  The exact
reference is computed at the *requested* h (= 0 for those rows), via
brute force when n <= EXACT_MAX_N -- so the relative error at h = 0
includes both the Taylor truncation error and the bias of running at
h_eff = 1/n instead.

Output: data/log_z_taylor.csv
  columns: graph_id, n, h, h_eff, beta, method, m,
           log_Z, log_Z_exact, rel_err, error_bound, runtime_s
"""

from __future__ import annotations

import csv
import json
import os
import signal
import sys
import time
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from adapter import (
    effective_h, log_z_error_bound, exact_log_Z,
    taylor_coefficients_timed, log_Z_from_coeffs,
)


BETAS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 5.0]
H_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0]
GRAPHS_TO_RUN = {"n16_graph0", "n16_graph1", "n16_graph2", "n16_graph3",
                 "n30_graph0", "n30_graph1", "n40_graph0", "n40_graph1",
                 "n50_graph0", "n50_graph1",
                 "n60_graph0", "n60_graph1"}
M_VALUES_SMALL = [2, 4, 6, 8, 10]   # naive on n <= 20
M_VALUES_LARGE = [2, 4, 6]          # insects on n > 20 (m=8 already minutes)
# Naive is cheap on small graphs (C(n,m)*|E| with small n) but blows up at
# large n; insects scales poly(n) for fixed m on bounded-degree graphs but
# is dominated by naive when n is small.  So we pick one per graph size:
SMALL_N_CUTOFF = 20
METHODS_SMALL = ["naive"]           # n <= SMALL_N_CUTOFF
METHODS_LARGE = ["insects"]         # n >  SMALL_N_CUTOFF
PER_CELL_TIMEOUT_S = 60.0
EXACT_MAX_N = 20

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GRAPHS_SUBDIR = os.path.join(DATA_DIR, "graphs")
OUT_CSV = os.path.join(DATA_DIR, "log_z_taylor.csv")


def load_graph(graph_id: str) -> Tuple[List[Tuple[int, int]], int]:
    with open(os.path.join(GRAPHS_SUBDIR, f"{graph_id}.json")) as f:
        gj = json.load(f)
    return [tuple(e) for e in gj["edges"]], gj["n"]


def _sort_key(gid: str) -> Tuple[int, int]:
    n_part, g_part = gid.split("_")
    return int(n_part[1:]), int(g_part.replace("graph", ""))


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _existing_cells(path: str) -> set:
    """Set of (graph_id, h, beta, method, m) tuples already in the CSV."""
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.add((row["graph_id"], float(row["h"]), float(row["beta"]),
                     row["method"], int(row["m"])))
    return out


def main():
    graph_ids = sorted(
        (fn[:-len(".json")] for fn in os.listdir(GRAPHS_SUBDIR)
         if fn.endswith(".json")),
        key=_sort_key,
    )
    graph_ids = [g for g in graph_ids if g in GRAPHS_TO_RUN]
    print(f"Found {len(graph_ids)} graphs.")

    signal.signal(signal.SIGALRM, _alarm)

    done = _existing_cells(OUT_CSV)
    write_header = not os.path.exists(OUT_CSV)
    mode = "w" if write_header else "a"
    with open(OUT_CSV, mode, newline="") as fout:
        w = csv.writer(fout)
        if write_header:
            w.writerow(["graph_id", "n", "h", "h_eff", "beta", "method", "m",
                        "log_Z", "log_Z_exact", "rel_err", "error_bound",
                        "runtime_s"])
        for graph_id in graph_ids:
            edges, n = load_graph(graph_id)
            if n <= SMALL_N_CUTOFF:
                methods, m_values = METHODS_SMALL, M_VALUES_SMALL
            else:
                methods, m_values = METHODS_LARGE, M_VALUES_LARGE
            print(f"\n=== {graph_id}  (n={n}, |E|={len(edges)}, "
                  f"methods={methods}, m={m_values}) ===")
            for h in H_VALUES:
                h_eff = effective_h(h, n)
                for beta in BETAS:
                    if n <= EXACT_MAX_N:
                        lz_exact = exact_log_Z(edges, n, beta, h)
                    else:
                        lz_exact = float("nan")
                    for method in methods:
                        wanted_m = [m for m in m_values
                                    if (graph_id, h, beta, method, m) not in done]
                        if not wanted_m:
                            continue
                        max_m = max(wanted_m)
                        signal.alarm(int(PER_CELL_TIMEOUT_S))
                        try:
                            coeffs, cum_times = taylor_coefficients_timed(
                                edges, n, beta, h, max_m, method=method)
                            ok = True
                        except _Timeout:
                            ok = False
                            print(f"   ! {method} max_m={max_m}: timeout")
                        except Exception as exc:
                            ok = False
                            print(f"   ! {method} max_m={max_m}: {exc}")
                        finally:
                            signal.alarm(0)
                        for m in wanted_m:
                            this_bound = log_z_error_bound(n, beta, h, m)
                            if ok:
                                try:
                                    lz = log_Z_from_coeffs(
                                        coeffs, edges, n, beta, h, m)
                                    lz_str = f"{lz:.6f}"
                                    if lz_exact == lz_exact:
                                        rel = abs(lz - lz_exact) / max(
                                            abs(lz_exact), 1e-12)
                                        rel_str = f"{rel:.6e}"
                                    else:
                                        rel_str = "nan"
                                    dt = cum_times[m]
                                except Exception as exc:
                                    lz_str = "nan"
                                    rel_str = "nan"
                                    dt = 0.0
                                    print(f"   ! {method} m={m}: {exc}")
                            else:
                                lz_str = "nan"
                                rel_str = "nan"
                                dt = 0.0
                            w.writerow([graph_id, n, h, h_eff, beta, method,
                                        m, lz_str,
                                        f"{lz_exact:.6f}" if lz_exact == lz_exact else "nan",
                                        rel_str, f"{this_bound:.6e}",
                                        f"{dt:.4f}"])
                    fout.flush()
                print(f"  h={h:.1f}  done")
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
