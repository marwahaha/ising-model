"""Jerrum-Sinclair FPRAS for the ferromagnetic Ising partition function.

Implements the chain MC_Ising and the FPRAS of
  Jerrum & Sinclair, "Polynomial-time approximation algorithms for the Ising
  model" (extended abstract), ICALP 1990, Springer LNCS 443, pp. 462-475
  (the file at /Users/macbookpro/Desktop/BFb0032051.pdf).

Notation follows the paper.

Hamiltonian (eq 1):
    H(sigma) = - sum_{(i,j) in E} V_ij sigma_i sigma_j - B sum_k sigma_k.

High-temperature expansion (eqs 2-5):
    Z(V_ij, B, beta) = A * Z'(mu),
    A      = (2 cosh(beta B))^n * prod_{(i,j) in E} cosh(beta V_ij)        (eq 4)
    lambda_ij = tanh(beta V_ij),  mu = tanh(beta B)                       (eq 3)
    Z'(mu) = sum_{X subset E} mu^{|odd(X)|} * prod_{(i,j) in X} lambda_ij (eqs 5,6)

The subgraphs-world Markov chain (Section 4):
  State space   Omega = 2^E  (all spanning subgraphs of E; |Omega| = 2^|E|).
  Stationary    pi(X) = w(X) / Z'(mu)  where  w(X) = mu^{|odd(X)|} prod lambda_ij.
  Transitions   from X:
    1. with probability 1/2 set X' = X;                       (laziness)
    2. else pick e in E uniformly, let Y = X XOR {e};
    3. if w(Y) >= w(X) set X' = Y;
       else set X' = Y with probability w(Y)/w(X), else X.
  This is single-edge-toggle Metropolis with proposal density 1/(2m).

The FPRAS for Z (Section 3, eqs 10-12):
  Z'(1) = prod_{(i,j) in E} (1 + lambda_ij)                              (eq 9)
  Schedule  mu_0 = 1,  mu_k = (n-k)/n for 1 <= k <= r,  mu_{r+1} = tanh(beta B),
  where r is the largest k with (n-k)/n > tanh(beta B); step size 1/n.    (eq 11)
  For consecutive (mu_k, mu_{k+1}) with mu_k > mu_{k+1} the importance ratio
    Z'(mu_{k+1}) / Z'(mu_k) = E_{pi_{mu_k}}[ (mu_{k+1}/mu_k)^{|odd(X)|} ]   (page 468)
  has integrand in [0, 1] (since mu_{k+1} < mu_k); telescope and multiply.
  Final: Z = A * Z'(1) * prod_k [Z'(mu_{k+1}) / Z'(mu_k)].                (eq 12)

This module also provides brute-force Z'(mu) by subgraph enumeration and a
brute-force log Z by spin enumeration, both for tests and small-graph
sanity checks.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------- the chain -----------------------------

class SubgraphsChain:
    """MC_Ising of Jerrum-Sinclair 1990, Section 4.

    State X ⊆ E represented by per-edge presence bits and per-vertex degree.
    At every step, with probability 1/2 the chain stays (self-loop); else it
    proposes toggling a uniformly chosen edge and accepts by Metropolis.
    """

    def __init__(self,
                 edges: Sequence[Tuple[int, int]],
                 n_vertices: int,
                 edge_lambdas: Sequence[float],
                 mu: float,
                 rng: Optional[random.Random] = None,
                 lazy: bool = True):
        if len(edges) != len(edge_lambdas):
            raise ValueError("edges and edge_lambdas length mismatch")
        self.edges: List[Tuple[int, int]] = [tuple(e) for e in edges]
        self.m = len(self.edges)
        self.n_vertices = int(n_vertices)
        self.lambdas: List[float] = [float(l) for l in edge_lambdas]
        self._mu: float = float(mu)
        self.rng = rng if rng is not None else random.Random()
        self.lazy = bool(lazy)
        # state
        self.in_X: List[bool] = [False] * self.m
        self.deg: List[int] = [0] * self.n_vertices
        self.n_odd: int = 0

    # --- accessors ---

    @property
    def mu(self) -> float:
        return self._mu

    def set_mu(self, mu: float) -> None:
        self._mu = float(mu)

    def reset_empty(self) -> None:
        """Reset to X = empty (always |odd|=0, w = mu^0 = 1)."""
        self.in_X = [False] * self.m
        self.deg = [0] * self.n_vertices
        self.n_odd = 0

    def odd_count(self) -> int:
        return self.n_odd

    def size(self) -> int:
        return sum(self.in_X)

    # --- single step of MC_Ising ---

    def step(self) -> bool:
        """One step of the chain.  Returns True iff the state changed.

        Implements the rule on page 471:
          1. with prob 1/2 set X' = X       (laziness; aperiodicity device)
          2. else pick edge e uniformly, Y = X XOR {e}
          3. accept iff w(Y) >= w(X), else with prob w(Y)/w(X)
        """
        if self.lazy and self.rng.random() < 0.5:
            return False
        i = self.rng.randrange(self.m)
        u, v = self.edges[i]
        was_in = self.in_X[i]
        sign = -1 if was_in else +1
        new_deg_u = self.deg[u] + sign
        new_deg_v = self.deg[v] + sign
        # Delta|odd|: +1 if endpoint becomes odd, -1 if becomes even.
        delta_odd = ((1 if (new_deg_u & 1) else -1)
                     + (1 if (new_deg_v & 1) else -1))
        # w(Y)/w(X) = lambda_e^sign * mu^delta_odd.
        lam = self.lambdas[i]
        mu_ = self._mu
        # Numerical care for mu == 0:
        if delta_odd > 0:
            if mu_ == 0.0:
                # Y has more odd vertices but mu = 0 ⇒ w(Y) = 0 ⇒ never accept.
                return False
            mu_pow = mu_ ** delta_odd
        elif delta_odd < 0:
            if mu_ == 0.0:
                # Y has strictly fewer odd vertices; w(X) might be 0 already
                # (then we are sampling from a degenerate measure).  If X has
                # |odd| > 0 and mu = 0 we still need to allow leaving — but
                # the stationary measure puts no mass on such X.  Treat
                # ratio = +inf, always accept (subject to lambda factor).
                mu_pow = float("inf")
            else:
                mu_pow = mu_ ** delta_odd  # > 1
        else:
            mu_pow = 1.0
        if sign == +1:
            ratio = lam * mu_pow  # may be inf
        else:
            # Removing an edge: ratio = (1/lam) * mu^delta_odd.
            if lam == 0.0:
                ratio = float("inf")
            else:
                ratio = mu_pow / lam
        if math.isinf(ratio) or ratio >= 1.0:
            accept = True
        else:
            accept = self.rng.random() < ratio
        if not accept:
            return False
        self.in_X[i] = not was_in
        self.deg[u] = new_deg_u
        self.deg[v] = new_deg_v
        self.n_odd += delta_odd
        return True

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()


# ----------------------- the partition-function FPRAS -----------------------

def schedule_mu(n: int, mu_target: float, step_n_mult: int = 1) -> List[float]:
    """Return the schedule (mu_0, mu_1, ..., mu_{r+1}) of eqs (10)-(11):
       mu_0 = 1,
       mu_k = (denom - k)/denom  for 1 <= k <= r,
       mu_{r+1} = mu_target,
    where denom = step_n_mult * n and r is the largest integer with
    (denom - r)/denom > mu_target.

    `step_n_mult` makes the schedule step size 1/(step_n_mult * n):
      = 1 (default) is the paper's 1/n step ~ n segments,
      > 1 is a finer schedule (smaller per-link gap, more segments).
    Each individual ratio Z'(mu_{k+1})/Z'(mu_k) lives in
    ( ((d-1)/d)^|odd|, 1 ] which is bounded away from 0 for all
    step_n_mult >= 1; coarser (step_n_mult < 1) would lose that.
    """
    if not (0.0 <= mu_target <= 1.0):
        raise ValueError("mu_target must be in [0, 1]")
    if mu_target >= 1.0:
        return [1.0]
    step_n_mult = max(1, int(step_n_mult))
    denom = step_n_mult * n
    r = int(math.ceil(denom * (1.0 - mu_target))) - 1
    r = max(r, 0)
    sched = [(denom - k) / denom for k in range(r + 1)]
    sched.append(mu_target)
    return sched


def _log_A(n: int, V_arr: np.ndarray, beta: float, B: float) -> float:
    """log of the scaling factor A from eq (4)."""
    return (n * math.log(2.0 * math.cosh(beta * B))
            + float(np.log(np.cosh(beta * V_arr)).sum()))


def estimate_log_Z(
    edges: Sequence[Tuple[int, int]],
    n: int,
    beta: float,
    B: float,
    V_ij: Optional[Sequence[float]] = None,
    burnin: int = 5_000,
    samples_per_step: int = 20_000,
    step_n_mult: int = 1,
    rng: Optional[random.Random] = None,
    verbose: bool = False,
) -> float:
    """Jerrum-Sinclair FPRAS estimate of log Z(V_ij, B, beta).

    Implements Steps 1-3 of the paper.  V_ij defaults to all-ones (J = 1).
    `step_n_mult` refines the mu schedule (see schedule_mu); =1 is the
    paper's 1/n step size.
    """
    rng = rng if rng is not None else random.Random()
    if V_ij is None:
        V_ij = [1.0] * len(edges)
    V_arr = np.asarray(V_ij, dtype=np.float64)
    lambdas = [math.tanh(beta * v) for v in V_ij]
    mu_target = math.tanh(beta * B)

    # Step 1: A and Z'(1).
    log_A = _log_A(n, V_arr, beta, B)
    log_Zp_1 = sum(math.log1p(l) for l in lambdas)  # log prod (1 + lambda_ij)

    # Step 2: schedule + ratio estimates.
    sched = schedule_mu(n, mu_target, step_n_mult=step_n_mult)
    if verbose:
        print(f"  schedule (length {len(sched)}): mu_0={sched[0]} ... "
              f"mu_{{{len(sched)-1}}}={sched[-1]}")

    chain = SubgraphsChain(edges, n, lambdas, sched[0], rng=rng)

    log_Z = log_A + log_Zp_1
    for k in range(len(sched) - 1):
        mu_k = sched[k]
        mu_kp1 = sched[k + 1]
        chain.set_mu(mu_k)
        chain.run(burnin)
        ratio_sum = 0.0
        for _ in range(samples_per_step):
            chain.step()
            odd = chain.n_odd
            if mu_k > 0.0:
                f = (mu_kp1 / mu_k) ** odd
            else:
                f = 1.0 if odd == 0 else 0.0
            ratio_sum += f
        ratio = ratio_sum / samples_per_step
        if ratio <= 0.0:
            raise RuntimeError(
                f"ratio estimate at step k={k} is non-positive: {ratio}. "
                "Increase samples_per_step.")
        log_Z += math.log(ratio)
        if verbose:
            print(f"    step {k:3d}: mu {mu_k:.6f} -> {mu_kp1:.6f}  "
                  f"ratio={ratio:.6f}  log_Z={log_Z:.5f}")
    return log_Z


def estimate_log_Z_trace(
    edges: Sequence[Tuple[int, int]],
    n: int,
    beta: float,
    B: float,
    V_ij: Optional[Sequence[float]] = None,
    burnin: int = 5_000,
    samples_per_step: int = 20_000,
    num_log_samples: int = 60,
    rng: Optional[random.Random] = None,
) -> Tuple[float, List[Tuple[int, float]]]:
    """Like estimate_log_Z but also returns the running log Z estimate at
    `num_log_samples` log-spaced step counts (cumulative inner-chain steps
    across burn-ins and sample phases of every schedule segment).

    During a segment's burn-in the running estimate is the committed total
    so far (the current segment contributes 0); during the sample phase the
    estimate adds log(running mean of f_k(X)) for the current segment.

    Returns (final_log_Z, [(total_step, log_Z_running), ...]).
    """
    rng = rng if rng is not None else random.Random()
    if V_ij is None:
        V_ij = [1.0] * len(edges)
    V_arr = np.asarray(V_ij, dtype=np.float64)
    lambdas = [math.tanh(beta * v) for v in V_ij]
    mu_target = math.tanh(beta * B)

    log_A = _log_A(n, V_arr, beta, B)
    log_Zp_1 = sum(math.log1p(l) for l in lambdas)

    sched = schedule_mu(n, mu_target)
    n_segments = len(sched) - 1
    if n_segments <= 0:
        final = log_A + log_Zp_1
        return final, [(0, final)]

    total_steps = n_segments * (burnin + samples_per_step)
    record_steps = np.unique(np.round(
        np.geomspace(1, total_steps, num=num_log_samples)
    ).astype(int))
    record_set = set(int(s) for s in record_steps)

    chain = SubgraphsChain(edges, n, lambdas, sched[0], rng=rng)

    log_Z_committed = log_A + log_Zp_1
    trace: List[Tuple[int, float]] = []
    total_t = 0

    for k in range(n_segments):
        mu_k = sched[k]
        mu_kp1 = sched[k + 1]
        chain.set_mu(mu_k)
        for _ in range(burnin):
            chain.step()
            total_t += 1
            if total_t in record_set:
                trace.append((total_t, log_Z_committed))
        ratio_sum = 0.0
        n_drawn = 0
        for _ in range(samples_per_step):
            chain.step()
            odd = chain.n_odd
            if mu_k > 0.0:
                f = (mu_kp1 / mu_k) ** odd
            else:
                f = 1.0 if odd == 0 else 0.0
            ratio_sum += f
            n_drawn += 1
            total_t += 1
            if total_t in record_set:
                rm = ratio_sum / n_drawn
                lz = log_Z_committed + math.log(rm) if rm > 0.0 else float("nan")
                trace.append((total_t, lz))
        ratio = ratio_sum / samples_per_step
        if ratio <= 0.0:
            raise RuntimeError(
                f"ratio estimate at step k={k} is non-positive: {ratio}. "
                "Increase samples_per_step.")
        log_Z_committed += math.log(ratio)
    return log_Z_committed, trace


# ------------------------- brute force (for tests) -------------------------

def brute_force_Z_prime(edges: Sequence[Tuple[int, int]], n: int,
                        lambdas: Sequence[float], mu: float) -> float:
    """Z'(mu) = sum_{X ⊆ E} mu^{|odd(X)|} * prod_{e in X} lambda_e.

    Enumerates 2^|E| subgraphs; only tractable for |E| <= ~26.
    """
    m = len(edges)
    if m > 26:
        raise ValueError("brute force only viable for |E| <= 26")
    total = 0.0
    for s in range(1 << m):
        deg = [0] * n
        prod = 1.0
        bit = 1
        for k in range(m):
            if s & bit:
                u, v = edges[k]
                deg[u] += 1
                deg[v] += 1
                prod *= lambdas[k]
            bit <<= 1
        n_odd = sum(1 for d in deg if d & 1)
        if mu == 0.0:
            if n_odd == 0:
                total += prod
        else:
            total += prod * (mu ** n_odd)
    return total


def exact_log_Z(edges: Sequence[Tuple[int, int]], n: int,
                beta: float, B: float,
                V_ij: Optional[Sequence[float]] = None) -> float:
    """log Z by brute force over 2^n spin configurations."""
    if V_ij is None:
        V_ij = [1.0] * len(edges)
    V_arr = np.asarray(V_ij, dtype=np.float64)
    idx = np.arange(1 << n, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(n)[None, :]) & 1
    spins = 2 * bits - 1
    i_arr = np.array([e[0] for e in edges])
    j_arr = np.array([e[1] for e in edges])
    bond_term = -((spins[:, i_arr] * spins[:, j_arr]) * V_arr).sum(axis=1)
    E_arr = bond_term.astype(np.float64) - B * spins.sum(axis=1)
    log_w = -beta * E_arr
    m = log_w.max()
    return float(m + math.log(np.exp(log_w - m).sum()))
