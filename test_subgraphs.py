"""Tests for the Jerrum-Sinclair subgraphs-world chain and FPRAS.

The implementation in subgraphs.py follows the paper
  Jerrum & Sinclair, "Polynomial-time approximation algorithms for the Ising
  model" (extended abstract), ICALP 1990 (PDF: BFb0032051.pdf).

Tests, in order:

  1.  high-temperature expansion identity
        Z(V_ij, B, beta) = A * Z'(tanh(beta B))
      checked by computing both sides on small graphs.  Should hold to ~1e-10.
  2.  brute-force Z'(mu) vs hand-computed closed-forms on small graphs.
  3.  schedule_mu correctness vs the (n-k)/n recipe in eq (11).
  4.  Stationarity of MC_Ising on K3 (Markov chain visit frequencies match
      pi(X) = mu^{|odd(X)|} prod_lambda / Z'(mu)).
  5.  FPRAS log Z vs exact, averaged over K seeds (K-seed mean must lie within
      3 standard errors of the exact value -- the principled test of an
      unbiased estimator).
  6.  Same as 5 with non-zero external field B.

Each test prints a labelled PASS/FAIL line; failures raise.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Callable, List, Tuple

import networkx as nx

from subgraphs import (
    SubgraphsChain,
    brute_force_Z_prime,
    estimate_log_Z,
    exact_log_Z,
    schedule_mu,
    _log_A,
)
import numpy as np


def _check(name: str, ok: bool, *details) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}", *details)
    if not ok:
        raise AssertionError(f"{name} failed: {details}")


# ---------------------- Test 1: identity Z = A * Z'(tanh beta B) ----------------------

def test_1_identity():
    print("\n--- Test 1: identity Z = A * Z'(tanh beta B) ---")
    cases = [
        ("triangle K3", [(0, 1), (1, 2), (0, 2)], 3),
        ("4-cycle",     [(0, 1), (1, 2), (2, 3), (3, 0)], 4),
        ("K4",          [(i, j) for i in range(4) for j in range(i+1, 4)], 4),
        ("path P5",     [(0, 1), (1, 2), (2, 3), (3, 4)], 5),
        ("3-reg n=6",   list(nx.random_regular_graph(3, 6, seed=0).edges()), 6),
    ]
    for name, edges, n in cases:
        for beta, B in [(0.4, 0.0), (0.8, 0.0), (0.6, 0.3), (1.0, 1.0)]:
            lambdas = [math.tanh(beta)] * len(edges)
            mu = math.tanh(beta * B)
            Zp = brute_force_Z_prime(edges, n, lambdas, mu)
            log_A_v = _log_A(n, np.ones(len(edges)), beta, B)
            log_Z_via = log_A_v + math.log(Zp)
            log_Z_true = exact_log_Z(edges, n, beta, B)
            err = abs(log_Z_via - log_Z_true)
            _check(f"{name:12s}  beta={beta:.2f}  B={B:.2f}  |Δlog Z|={err:.2e}",
                   err < 1e-9)


# ---------------------- Test 2: closed-form Z'(mu) ----------------------

def test_2_closed_form_Zp():
    """Z'(mu) hand-computed for small graphs."""
    print("\n--- Test 2: closed-form Z'(mu) vs brute force ---")
    cases: List[Tuple[str, list, int, Callable[[float, float], float]]] = [
        # single edge: subgraphs {}, {edge}; Z' = 1 + lambda * mu^2
        ("single edge", [(0, 1)], 2,
         lambda l, m: 1.0 + l * m**2),
        # K3 = triangle (3 edges).  Subsets:
        #   empty (1, |odd|=0)
        #   1 edge x 3 (|odd|=2)
        #   2 edges x 3 (path; |odd|=2)
        #   3 edges x 1 (cycle; Eulerian)
        ("K3",
         [(0, 1), (1, 2), (0, 2)], 3,
         lambda l, m: 1.0 + 3*l*m**2 + 3*l**2*m**2 + l**3),
        # 4-cycle: 4 edges.
        #   empty: 1
        #   1 edge x 4: lambda mu^2
        #   2 adjacent edges x 4 (a P_3, |odd|=2 at endpoints): l^2 mu^2
        #   2 disjoint edges x 2 (matching, all 4 endpoints odd): l^2 mu^4
        #   3 edges x 4 (P_4 path, |odd|=2): l^3 mu^2
        #   4 edges x 1: l^4 (Eulerian)
        ("4-cycle",
         [(0, 1), (1, 2), (2, 3), (3, 0)], 4,
         lambda l, m: 1.0 + 4*l*m**2 + 4*l**2*m**2 + 2*l**2*m**4 + 4*l**3*m**2 + l**4),
    ]
    for name, edges, n, closed in cases:
        m = len(edges)
        for lam_val in (0.1, 0.3, 0.7):
            for mu in (0.0, 0.3, 0.7, 1.0):
                lambdas = [lam_val] * m
                got = brute_force_Z_prime(edges, n, lambdas, mu)
                want = closed(lam_val, mu)
                err = abs(got - want)
                _check(f"{name:12s}  lambda={lam_val:.2f}  mu={mu:.2f}  "
                       f"brute={got:.6f}  closed={want:.6f}",
                       err < 1e-10)


# ---------------------- Test 3: schedule_mu correctness ----------------------

def test_3_schedule():
    """schedule_mu(n, mu_target) returns mu_0=1, mu_k=(n-k)/n for 1<=k<=r,
    mu_{r+1}=mu_target, where (n-r)/n > mu_target.  See eq (11)."""
    print("\n--- Test 3: schedule_mu vs eq (11) ---")
    for n in (6, 10, 16):
        for mu_target in (0.0, 0.05, 0.4, 0.55, 0.9):
            s = schedule_mu(n, mu_target)
            ok = (s[0] == 1.0 and s[-1] == mu_target)
            for k in range(1, len(s) - 1):
                if abs(s[k] - (n - k) / n) > 1e-12:
                    ok = False
                    break
            # second-to-last must be strictly > mu_target
            ok = ok and (len(s) >= 2 and s[-2] > mu_target)
            # step size in interior is exactly 1/n
            for k in range(len(s) - 2):
                if abs((s[k] - s[k+1]) - 1.0 / n) > 1e-12:
                    ok = False
                    break
            _check(f"n={n:3d}  mu_target={mu_target:.3f}  "
                   f"len(sched)={len(s)}  sched=[{s[0]}, {s[1] if len(s)>1 else None}, ..., {s[-1]}]",
                   ok)


# ---------------------- Test 4: stationarity on K3 ----------------------

def test_4_stationarity():
    """Run MC_Ising on K3 at fixed (lambda, mu), count visits to all 8 states,
    and compare empirical frequencies to pi(X) = w(X)/Z'(mu)."""
    print("\n--- Test 4: stationarity of MC_Ising on K3 ---")
    edges = [(0, 1), (1, 2), (0, 2)]
    n = 3
    lambdas = [0.6, 0.6, 0.6]
    mu = 0.4
    Zp = brute_force_Z_prime(edges, n, lambdas, mu)
    rng = random.Random(7)
    chain = SubgraphsChain(edges, n, lambdas, mu, rng=rng, lazy=True)
    chain.run(20_000)  # burn-in
    n_samples = 800_000
    counts: Counter = Counter()
    for _ in range(n_samples):
        chain.step()
        counts[tuple(chain.in_X)] += 1

    # Analytical pi(X).
    def w(state):
        deg = [0, 0, 0]
        prod = 1.0
        for k, present in enumerate(state):
            if present:
                u, v = edges[k]
                deg[u] += 1
                deg[v] += 1
                prod *= lambdas[k]
        n_odd = sum(1 for d in deg if d & 1)
        return prod * (mu ** n_odd if mu > 0 else (1.0 if n_odd == 0 else 0.0))

    print(f"     state                p_true   p_empirical   |diff|")
    tv = 0.0
    for state in sorted(counts):
        p_true = w(state) / Zp
        p_emp = counts[state] / n_samples
        tv += abs(p_true - p_emp)
        print(f"     {state}    {p_true:.5f}   {p_emp:.5f}    {abs(p_true-p_emp):.2e}")
    tv *= 0.5
    _check(f"TV(empirical, pi) = {tv:.4f}", tv < 0.01)


# ---------------------- Test 5: FPRAS log Z (B=0) ----------------------

def test_5_fpras_no_field():
    """JS FPRAS log Z vs exact at B=0.  K-seed mean must lie within 3 SE of
    the exact value."""
    print("\n--- Test 5: JS FPRAS log Z vs exact (B=0), K-seed average ---")
    cases = [
        ("triangle K3", [(0, 1), (1, 2), (0, 2)], 3),
        ("4-cycle",     [(0, 1), (1, 2), (2, 3), (3, 0)], 4),
        ("K4",          [(i, j) for i in range(4) for j in range(i+1, 4)], 4),
        ("3-reg n=6",   list(nx.random_regular_graph(3, 6, seed=0).edges()), 6),
        ("3-reg n=8",   list(nx.random_regular_graph(3, 8, seed=0).edges()), 8),
    ]
    K_SEEDS = 6
    for name, edges, n in cases:
        for beta in (0.3, 0.6, 1.0):
            log_Z_true = exact_log_Z(edges, n, beta, 0.0)
            est = []
            for seed in range(K_SEEDS):
                rng = random.Random(1000 + seed)
                est.append(estimate_log_Z(
                    edges, n, beta, B=0.0,
                    burnin=5_000, samples_per_step=15_000,
                    rng=rng,
                ))
            mean = sum(est) / len(est)
            var = sum((e - mean) ** 2 for e in est) / (len(est) - 1)
            se = math.sqrt(var / len(est))
            err_mean = mean - log_Z_true
            _check(
                f"{name:12s}  beta={beta:.2f}  exact={log_Z_true:.4f}  "
                f"mean={mean:.4f}  SE={se:.4f}  Δ={err_mean:+.4f}  "
                f"|Δ|/SE={abs(err_mean)/se:.2f}σ",
                abs(err_mean) < 3 * se + 0.01,
            )


# ---------------------- Test 6: FPRAS log Z (B != 0) ----------------------

def test_6_fpras_with_field():
    print("\n--- Test 6: JS FPRAS log Z with field, K-seed average ---")
    cases = [
        ("K3",      [(0, 1), (1, 2), (0, 2)], 3),
        ("4-cycle", [(0, 1), (1, 2), (2, 3), (3, 0)], 4),
        ("K4",      [(i, j) for i in range(4) for j in range(i+1, 4)], 4),
    ]
    K_SEEDS = 6
    for name, edges, n in cases:
        for beta, B in [(0.4, 0.2), (0.6, 0.5), (0.8, 1.0)]:
            log_Z_true = exact_log_Z(edges, n, beta, B)
            est = []
            for seed in range(K_SEEDS):
                rng = random.Random(2000 + seed)
                est.append(estimate_log_Z(
                    edges, n, beta, B=B,
                    burnin=5_000, samples_per_step=15_000,
                    rng=rng,
                ))
            mean = sum(est) / len(est)
            var = sum((e - mean) ** 2 for e in est) / (len(est) - 1)
            se = math.sqrt(var / len(est))
            err_mean = mean - log_Z_true
            _check(
                f"{name:10s}  beta={beta:.2f}  B={B:.2f}  exact={log_Z_true:.4f}  "
                f"mean={mean:.4f}  SE={se:.4f}  Δ={err_mean:+.4f}  "
                f"|Δ|/SE={abs(err_mean)/se:.2f}σ",
                abs(err_mean) < 3 * se + 0.01,
            )


if __name__ == "__main__":
    test_1_identity()
    test_2_closed_form_Zp()
    test_3_schedule()
    test_4_stationarity()
    test_5_fpras_no_field()
    test_6_fpras_with_field()
    print("\nAll tests passed.")
