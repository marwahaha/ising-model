"""Adapter between the project's physical (beta, h) convention and the LSS
paper's (beta_LSS, lambda) convention used by taylor_lss.py.

Project Hamiltonian:   H(sigma) = - sum_<u,v> sigma_u sigma_v - h sum_v sigma_v
Boltzmann weight:      exp(-beta * H)  -- so beta is inverse temperature.

LSS edge / vertex activity:
    beta_LSS = exp(-2 * beta)
    lambda   = exp(2 * beta * h_eff)

Relation of partition functions:
    log Z_phys = beta * |E|  -  beta * h_eff * n  +  log Z_LSS(lambda)

For h = 0 we use the Jerrum-Sinclair trick of replacing h with h_eff = 1/n
to push lambda off the Lee-Yang circle.  Comparisons are done at h_eff,
not at the requested h = 0.
"""

from __future__ import annotations

import cmath
import math
import time
from itertools import combinations
from typing import Iterator, List, Tuple

import numpy as np

from taylor_core import (
    z_approximation, truncation_error_bound,
    z_polynomial_insects, z_polynomial_insects_iter,
    _coefficients_to_z_approx,
)


def effective_h(h: float, n: int) -> float:
    """h_eff = 1/n if h == 0 else h (JS field-anneal trick)."""
    return 1.0 / n if h == 0.0 else h


def lss_params(beta: float, h_eff: float) -> Tuple[complex, complex]:
    """Return (beta_LSS, lambda) corresponding to our (beta, h_eff)."""
    return complex(math.exp(-2.0 * beta)), complex(math.exp(2.0 * beta * h_eff))


def _z_polynomial_naive_np(n: int, edges: List[Tuple[int, int]],
                           beta: complex, max_degree: int) -> List[complex]:
    """Numpy-vectorised version of taylor_lss.z_polynomial_naive.  Builds an
    [n_subsets, n] boolean indicator matrix, computes the cut count per
    subset against the edge list in one shot.  ~30x faster than the pure-
    Python combinations loop on n=16."""
    M = min(max_degree, n)
    edges_arr = np.asarray(edges, dtype=np.int64)
    e_u = edges_arr[:, 0]
    e_v = edges_arr[:, 1]
    coeffs = [complex(0)] * (M + 1)
    coeffs[0] = complex(1)
    for k in range(1, M + 1):
        idx = np.fromiter(
            (v for S in combinations(range(n), k) for v in S),
            dtype=np.int64,
        ).reshape(-1, k)
        if idx.size == 0:
            continue
        bits = np.zeros((idx.shape[0], n), dtype=bool)
        rows = np.arange(idx.shape[0])
        for j in range(k):
            bits[rows, idx[:, j]] = True
        cut_count = (bits[:, e_u] ^ bits[:, e_v]).sum(axis=1)
        coeffs[k] = complex((beta ** cut_count.astype(np.complex128)).sum())
    return coeffs


def estimate_log_Z(edges: List[Tuple[int, int]], n: int,
                   beta: float, h: float, m: int,
                   method: str = "naive") -> float:
    """LSS Taylor estimate of log Z(beta, h_eff) at truncation order m,
    where h_eff = effective_h(h, n).  method ∈ {'naive', 'insects'}.

    Prefer ``taylor_coefficients_timed`` + ``log_Z_from_coeffs`` when you
    need several values of m for the same (graph, beta, h): that path
    computes coefficients once incrementally instead of from scratch at
    every m."""
    h_eff = effective_h(h, n)
    beta_lss, lam = lss_params(beta, h_eff)
    use_inverse = abs(lam) > 1
    target = (1 / lam) if use_inverse else lam
    if method == "naive":
        coeffs = _z_polynomial_naive_np(n, edges, beta_lss, m)
    elif method == "insects":
        coeffs = z_polynomial_insects(n, edges, beta_lss, m)
    else:
        raise ValueError(f"Unknown method: {method!r}")
    _, z_hat = _coefficients_to_z_approx(coeffs, target, m)
    if use_inverse:
        z_hat = (lam ** n) * z_hat
    log_z_lss = cmath.log(z_hat).real
    return beta * len(edges) - beta * h_eff * n + log_z_lss


def _naive_coeffs_timed(n: int, edges: List[Tuple[int, int]],
                        beta: complex, max_m: int
                        ) -> Iterator[Tuple[int, complex, float]]:
    """Incremental naive coefficient generator.  Yields (k, c_k, elapsed)
    where elapsed is the cumulative wall time to compute coeffs[0..k].
    c_0 = 1 is yielded at t = 0."""
    M = min(max_m, n)
    edges_arr = np.asarray(edges, dtype=np.int64)
    e_u = edges_arr[:, 0]
    e_v = edges_arr[:, 1]
    t0 = time.time()
    yield 0, complex(1), 0.0
    for k in range(1, M + 1):
        idx = np.fromiter(
            (v for S in combinations(range(n), k) for v in S),
            dtype=np.int64,
        ).reshape(-1, k)
        if idx.size == 0:
            c_k = complex(0)
        else:
            bits = np.zeros((idx.shape[0], n), dtype=bool)
            rows = np.arange(idx.shape[0])
            for j in range(k):
                bits[rows, idx[:, j]] = True
            cut_count = (bits[:, e_u] ^ bits[:, e_v]).sum(axis=1)
            c_k = complex((beta ** cut_count.astype(np.complex128)).sum())
        yield k, c_k, time.time() - t0


def _insects_coeffs_timed(n: int, edges: List[Tuple[int, int]],
                          beta: complex, max_m: int
                          ) -> Iterator[Tuple[int, complex, float]]:
    """Incremental insects coefficient generator.  Yields ``(k, c_k,
    elapsed)`` for k = 0..max_m, where ``elapsed`` is the cumulative
    wall time since the first yield.  Backed by
    :func:`taylor_core.z_polynomial_insects_iter`, which interleaves
    Newton's identities into the main DP loop so c_t is available
    immediately after a_table[t] is built."""
    t0 = time.time()
    for k, c_k in z_polynomial_insects_iter(n, edges, beta, max_m):
        yield k, c_k, 0.0 if k == 0 else (time.time() - t0)


def taylor_coefficients_timed(edges: List[Tuple[int, int]], n: int,
                              beta: float, h: float, max_m: int,
                              method: str = "naive"
                              ) -> Tuple[List[complex], List[float]]:
    """Compute Taylor coefficients up to max_m once, with per-k cumulative
    wall time.  Returns (coeffs, cum_times) each of length max_m+1.

    ``coeffs[k]`` is the polynomial coefficient at degree k (with the
    paper's normalisation, c_0 = 1).  ``cum_times[k]`` is the wall time
    to compute coeffs[0..k]."""
    h_eff = effective_h(h, n)
    beta_lss, _ = lss_params(beta, h_eff)
    if method == "naive":
        gen = _naive_coeffs_timed(n, edges, beta_lss, max_m)
    elif method == "insects":
        gen = _insects_coeffs_timed(n, edges, beta_lss, max_m)
    else:
        raise ValueError(f"Unknown method: {method!r}")
    coeffs: List[complex] = []
    cum_times: List[float] = []
    for _k, c_k, dt in gen:
        coeffs.append(c_k)
        cum_times.append(dt)
    return coeffs, cum_times


def log_Z_from_coeffs(coeffs: List[complex], edges: List[Tuple[int, int]],
                      n: int, beta: float, h: float, m: int) -> float:
    """Plug pre-computed coefficients into the LSS Taylor formula at
    truncation order m.  Cheap (O(m^2))."""
    h_eff = effective_h(h, n)
    _, lam = lss_params(beta, h_eff)
    use_inverse = abs(lam) > 1
    target = (1 / lam) if use_inverse else lam
    _, z_hat = _coefficients_to_z_approx(coeffs[:m + 1], target, m)
    if use_inverse:
        z_hat = (lam ** n) * z_hat
    log_z_lss = cmath.log(z_hat).real
    return beta * len(edges) - beta * h_eff * n + log_z_lss


def log_z_error_bound(n: int, beta: float, h: float, m: int) -> float:
    """Additive log-Z error bound (Lemma 2.1) at h_eff = effective_h(h, n)."""
    h_eff = effective_h(h, n)
    _, lam = lss_params(beta, h_eff)
    return truncation_error_bound(n, lam, m)


def exact_log_Z(edges: List[Tuple[int, int]], n: int,
                beta: float, h: float) -> float:
    """Brute-force log Z(beta, h) by spin enumeration -- at the *requested*
    h (no field-anneal substitution).  This is the reference against which
    LSS estimates are compared, so that the relative error at h = 0 captures
    both the Taylor truncation error and the bias of running at h_eff = 1/n
    rather than at the true h = 0.  Tractable up to n ~= 22."""
    idx = np.arange(1 << n, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(n)[None, :]) & 1
    spins = 2 * bits - 1
    i_arr = np.array([i for i, _ in edges])
    j_arr = np.array([j for _, j in edges])
    bond_sum = (spins[:, i_arr] * spins[:, j_arr]).sum(axis=1)
    E_arr = -bond_sum.astype(np.float64) - h * spins.sum(axis=1)
    log_w = -beta * E_arr
    mx = log_w.max()
    return float(mx + np.log(np.exp(log_w - mx).sum()))
