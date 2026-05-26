"""
Liu-Sinclair-Srivastava (LSS) / Barvinok-style Taylor approximation of the
ferromagnetic Ising partition function — two implementations.

Paper: "The Ising Partition Function: Zeros and Deterministic Approximation",
Liu, Sinclair, Srivastava (arXiv:1704.06493, 2017).

We use the paper's formulation. On a graph G = (V, E) with edge activity β
and vertex activity λ:

    Z(λ) = Σ_{S ⊆ V}  β^{|cut(S)|}  ·  λ^{|S|}
         = Σ_{σ ∈ {+,-}^V}  Π_{e ∈ E} φ_e(σ_e)  ·  λ^{|{v: σ(v)=+}|}

with φ_e(σ_e) = β if e is cut (endpoints differ), 1 otherwise (the paper's
normalization φ_e(-,...,-) = 1).  Equivalently, with physical couplings
β_phys = e^{-2J} and λ = e^{2h} relates this to the usual Hamiltonian
H(σ) = -J Σ σ_u σ_v - h Σ σ_v.

The Barvinok pipeline (Section 2 of the paper):
    1. Compute the first m+1 polynomial coefficients of Z(λ) = Σ c_i λ^i.
    2. Read off Z's derivatives at 0:    Z^(j)(0) = j! · c_j.
    3. Solve the triangular Leibniz system (eq. 5) for f^(j)(0), f = log Z.
    4. Form f_m(λ) = Σ f^(j)(0) λ^j / j!.
    5. Output Z̃ = exp(f_m(λ)) as the (1 ± ε) approximation.

By Lemma 2.1, if Lee-Yang holds (β ∈ [0, 1], |λ| ≠ 1), then
    |f(λ) - f_m(λ)|  ≤  n · |λ|^{m+1} / ((m+1)(1 - |λ|))
so m = O(log(n/ε) / log(1/|λ|)) suffices for additive ε in log Z.

The bottleneck is step 1.  Two methods:
  * 'naive'   — enumerate subsets of size ≤ m directly from the definition.
                Cost: O(n^m · |E|).  This is the paper's n^Ω(m) baseline
                (quasi-polynomial for m = O(log n)) — *not* a full 2^n sweep,
                since we only need the first m+1 polynomial coefficients of Z.
  * 'insects' — the Patel-Regts insect-counting DP of Section 3, polynomial in
                n for graphs of bounded degree when m = O(log n).

Both methods feed step 1 of the same pipeline and (modulo floating point)
return identical answers.

CLI:
    python taylor_lss.py --graph K5 --beta 0.5 --lam 0.4 --m 6 --method both
"""

from __future__ import annotations

import argparse
import cmath
import math
from collections import defaultdict
from itertools import combinations
from math import comb, factorial
from typing import Iterable


# ===================================================================
# 1.  Naive Z-polynomial (O(2^n))
# ===================================================================

def z_polynomial_naive(n: int, edges: list[tuple[int, int]],
                       beta: complex,
                       max_degree: int | None = None) -> list[complex]:
    """
    Return [c_0, c_1, ..., c_M] with  Z(λ) = Σ_i c_i λ^i,   c_i = Σ_{|S|=i} β^|cut(S)|.

    M defaults to n (the full polynomial, costing O(2^n · |E|) — exponential).
    When used as a Taylor-truncation subroutine, pass max_degree=m: the loop
    only enumerates subsets of size ≤ m, costing  Σ_{k≤m} C(n,k) · |E|
    = O(n^m · |E|).  This is the n^Ω(m) baseline mentioned just before
    Section 3 of the paper, which Patel-Regts then improves to poly(n).
    """
    M = n if max_degree is None else min(max_degree, n)
    coeffs = [0 + 0j] * (M + 1)
    for k in range(M + 1):
        for S in combinations(range(n), k):
            S_set = set(S)
            cut = sum(1 for (u, v) in edges if (u in S_set) ^ (v in S_set))
            coeffs[k] += beta ** cut
    return coeffs


# ===================================================================
# 2.  Patel-Regts insect machinery (Section 3 of the paper)
# ===================================================================

def _adj_list(n: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _enumerate_connected_subsets(adj: list[set[int]], n: int,
                                 max_size: int) -> set[frozenset[int]]:
    """
    Lemma 3.9: enumerate all S ⊆ V(G) with G[S] connected and |S| ≤ max_size.

    Inductive: T_t = T_{t-1} ∪ {S ∪ {v} : S ∈ T_{t-1}, v ∈ N(S) \\ S}.
    Connectedness is preserved because v is adjacent to some vertex in S.
    """
    if max_size <= 0:
        return set()
    layer: set[frozenset[int]] = {frozenset([v]) for v in range(n)}
    all_sets: set[frozenset[int]] = set(layer)
    for _ in range(2, max_size + 1):
        next_layer: set[frozenset[int]] = set()
        for S in layer:
            nbrs: set[int] = set()
            for v in S:
                nbrs |= adj[v]
            nbrs -= S
            for v in nbrs:
                next_layer.add(S | {v})
        layer = next_layer
        if not layer:
            break
        all_sets |= layer
    return all_sets


def _is_connected_induced(S: frozenset[int], adj: list[set[int]]) -> bool:
    """G[S] connected (treating S as a vertex set)."""
    if not S:
        return True
    start = next(iter(S))
    seen = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u in S and u not in seen:
                seen.add(u)
                stack.append(u)
    return seen == set(S)


def _mu_factory(edges: list[tuple[int, int]], beta: complex):
    """
    µ_S = (-1)^|S| · β^|cut_G(S)|  with caching on the subset.

    This is the coefficient of 1[H ↪ G] in e_{|H|}(G), for H = G+[S]; see
    eq. (10) of the paper specialized to graphs with the Ising weight.
    """
    cache: dict[frozenset[int], complex] = {}

    def mu(S: frozenset[int]) -> complex:
        v = cache.get(S)
        if v is not None:
            return v
        cut = sum(1 for (u, w) in edges if (u in S) ^ (w in S))
        v = ((-1) ** len(S)) * beta ** cut
        cache[S] = v
        return v

    return mu


def z_polynomial_insects_iter(n: int, edges: list[tuple[int, int]],
                              beta: complex, m: int):
    """Generator form of :func:`z_polynomial_insects`.  Yields ``(t, c_t)``
    for t = 0..m as each coefficient becomes available.  Interleaving
    Newton's identities into the main DP loop means c_t is ready as soon
    as ``a_table[t]`` is finished — useful for incremental timing or
    early termination."""
    if m < 0:
        raise ValueError("m must be ≥ 0")
    yield 0, 1 + 0j   # c_0 = 1
    if m == 0:
        return

    adj = _adj_list(n, edges)
    mu = _mu_factory(edges, beta)

    # (a) connected sub-insects of G of size ≤ m
    connected = _enumerate_connected_subsets(adj, n, m)

    # Bucket connected sets by size, for fast iteration in the recurrence.
    connected_by_size: dict[int, list[frozenset[int]]] = defaultdict(list)
    for S in connected:
        connected_by_size[len(S)].append(S)

    # (b) DP.  a_table[t][S] = a_S^(t) for connected S with |S| ≤ t.
    # By Lemma 3.3, a_S^(t) = 0 for disconnected S — we never store those.
    a_table: dict[int, dict[frozenset[int], complex]] = {0: {}}
    p: list[complex] = []   # p[t-1] = p_t
    e: list[complex] = []   # e[t-1] = e_t

    for t in range(1, m + 1):
        cur: dict[frozenset[int], complex] = {}
        a_table[t] = cur

        for s_size in range(1, t + 1):
            for S in connected_by_size.get(s_size, ()):
                val: complex = 0 + 0j

                # Boundary term in eq. (11): (-1)^{t-1} · t · µ_S · [|S| = t].
                if s_size == t:
                    val += ((-1) ** (t - 1)) * t * mu(S)

                # Sum over i = 1..t-1 of (-1)^{i-1} · [coeff of S in p_{t-i} e_i].
                # By Corollary 3.6 and eq. (14):
                #   coeff at S = Σ_{(S1,S2): S1∪S2=S, |S1|=i, G[S2] connected,
                #                            |S2| ≤ t-i}  µ_{S1} · a_{S2}^{(t-i)}
                S_list = list(S)
                for i in range(1, t):
                    a_prev = a_table[t - i]
                    inner: complex = 0 + 0j
                    for S1_tup in combinations(S_list, i):
                        S1 = frozenset(S1_tup)
                        S_minus = S - S1
                        if len(S_minus) > t - i:
                            continue  # S2 ⊇ S_minus already too big
                        max_extra = (t - i) - len(S_minus)
                        # S2 = S_minus ∪ T,  T ⊆ S1,  |T| ≤ max_extra.
                        S1_list = list(S1)
                        # We need µ_{S1} regardless of T.
                        mu_S1 = mu(S1)
                        if mu_S1 == 0:
                            continue
                        for t_size in range(min(max_extra, len(S1_list)) + 1):
                            for T_tup in combinations(S1_list, t_size):
                                S2 = S_minus | frozenset(T_tup)
                                a_S2 = a_prev.get(S2)
                                if a_S2:  # nonzero & present (so S2 connected)
                                    inner += mu_S1 * a_S2
                    val += ((-1) ** (i - 1)) * inner

                if val != 0:
                    cur[S] = val

        # (c) p_t = Σ_S a_S^(t)
        p_t = sum(cur.values())
        p.append(p_t)

        # (d) e_t via Newton's identities (eq. 11).
        #   e_t = (-1)^{t-1}/t · [p_t − Σ_{i=1}^{t-1} (-1)^{i-1} p_{t-i} e_i]
        s = p_t
        for i in range(1, t):
            s -= ((-1) ** (i - 1)) * p[t - i - 1] * e[i - 1]
        e_t = ((-1) ** (t - 1)) * s / t
        e.append(e_t)

        # (e) c_t = (-1)^t e_t.
        yield t, ((-1) ** t) * e_t


def z_polynomial_insects(n: int, edges: list[tuple[int, int]],
                         beta: complex, m: int) -> list[complex]:
    """
    Patel-Regts insect DP: return [c_0, c_1, ..., c_m] with
        Z(λ)  =  Σ_{i=0}^{n} (-1)^i e_i(G) λ^i,
    truncated at degree m; c_0 = 1.  Thin wrapper around
    :func:`z_polynomial_insects_iter` that materialises all coefficients
    into a list.

    Pipeline (paper Section 3.4):
        (a) Enumerate connected induced subgraphs S ⊆ V(G), |S| ≤ m.
        (b) DP for a_S^(t), the coefficient of 1[G+[S] ↪ G] in p_t :=
            Σ_j 1/r_j^t. Recurrence is eq. (14) of the paper.
        (c) p_t(G) = Σ_{S connected} a_S^(t).
        (d) Newton's identities (eq. 11) invert p_t ↔ e_t.
        (e) c_i = (-1)^i e_i, with e_0 = 1.
    """
    return [c for _, c in z_polynomial_insects_iter(n, edges, beta, m)]


# ===================================================================
# 3.  Shared pipeline:  polynomial coefficients → Taylor approximation
# ===================================================================

def evaluate_polynomial(coeffs: Iterable[complex], lam: complex) -> complex:
    """Horner evaluation of Σ_i c_i λ^i."""
    z = 0 + 0j
    for c in reversed(list(coeffs)):
        z = z * lam + c
    return z


def z_derivatives_at_zero(coeffs: list[complex], m: int) -> list[complex]:
    """Z^(j)(0) = j! · c_j for j = 0..m (zero past the polynomial degree)."""
    out: list[complex] = []
    for j in range(m + 1):
        cj = coeffs[j] if j < len(coeffs) else (0 + 0j)
        out.append(factorial(j) * cj)
    return out


def log_z_derivatives_at_zero(z_derivs: list[complex], m: int) -> list[complex]:
    """
    Triangular Leibniz system (eq. 5 of the paper):
        Z^(m) = Σ_{j=0}^{m-1} C(m-1, j) · Z^(j) · f^(m-j)   (at λ=0)
    With Z(0) = 1, the j=0 term isolates f^(m)(0).
    """
    if abs(z_derivs[0] - 1) > 1e-10:
        raise ValueError("Need Z(0) = 1 (paper normalization φ_e(-,...,-) = 1).")
    f: list[complex] = [0 + 0j]
    for k in range(1, m + 1):
        s = z_derivs[k]
        for j in range(1, k):
            s -= comb(k - 1, j) * z_derivs[j] * f[k - j]
        f.append(s)
    return f


def taylor_truncation(f_derivs: list[complex], lam: complex, m: int) -> complex:
    """f_m(λ) = Σ_{j=0}^{m} f^(j)(0) · λ^j / j!."""
    s = 0 + 0j
    pow_lam: complex = 1 + 0j
    fact = 1
    for j in range(m + 1):
        s += f_derivs[j] * pow_lam / fact
        pow_lam *= lam
        fact *= (j + 1)
    return s


def _coefficients_to_z_approx(coeffs: list[complex], lam: complex,
                              m: int) -> tuple[complex, complex]:
    """Return (f_m(λ), Z̃ = exp(f_m(λ)))."""
    z_d = z_derivatives_at_zero(coeffs, m)
    f_d = log_z_derivatives_at_zero(z_d, m)
    f_m = taylor_truncation(f_d, lam, m)
    return f_m, cmath.exp(f_m)


def z_approximation(n: int, edges: list[tuple[int, int]],
                    beta: complex, lam: complex, m: int,
                    method: str = "naive") -> complex:
    """
    Top-level approximator. Returns Z̃ ≈ Z(λ).

    method = 'naive'   : O(2^n) brute-force polynomial extraction.
    method = 'insects' : Patel-Regts insect DP, poly(n) for bounded degree
                         when m = O(log n).

    The Taylor series around λ = 0 converges only for |λ| < 1. For |λ| > 1
    we use the paper's λ ↔ 1/λ symmetry (ferromagnetic Ising has
    φ_e(σ) = φ_e(-σ), so Z(λ) = λ^n · Z(1/λ)); see the paragraph after
    eq. (3). |λ| = 1 is exactly the Lee-Yang circle — excluded.
    """
    if abs(lam) == 1:
        raise ValueError("|λ| = 1 is on the Lee-Yang circle (excluded by Theorem 1.1).")
    use_inverse = abs(lam) > 1
    target = (1 / lam) if use_inverse else lam

    if method == "naive":
        coeffs = z_polynomial_naive(n, edges, beta, max_degree=m)
    elif method == "insects":
        coeffs = z_polynomial_insects(n, edges, beta, m)
    else:
        raise ValueError(f"Unknown method: {method!r} (use 'naive' or 'insects')")
    _, z_hat = _coefficients_to_z_approx(coeffs, target, m)
    return (lam ** n) * z_hat if use_inverse else z_hat


def truncation_error_bound(n: int, lam: complex, m: int) -> float:
    """
    Lemma 2.1 additive bound on log Z, assuming Lee-Yang holds.
    For |λ| > 1 we apply the same bound with the |1/λ| < 1 substitution
    used inside z_approximation.
    """
    a = abs(lam)
    if a == 1:
        return float("inf")
    if a > 1:
        a = 1 / a
    return n * a ** (m + 1) / ((m + 1) * (1 - a))


# ===================================================================
# 4.  Graph builders
# ===================================================================

def cycle(n: int) -> tuple[int, list[tuple[int, int]]]:
    return n, [(i, (i + 1) % n) for i in range(n)]


def path(n: int) -> tuple[int, list[tuple[int, int]]]:
    return n, [(i, i + 1) for i in range(n - 1)]


def complete(n: int) -> tuple[int, list[tuple[int, int]]]:
    return n, [(i, j) for i in range(n) for j in range(i + 1, n)]


def grid(rows: int, cols: int) -> tuple[int, list[tuple[int, int]]]:
    n = rows * cols
    edges = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                edges.append((v, v + 1))
            if r + 1 < rows:
                edges.append((v, v + cols))
    return n, edges


def petersen() -> tuple[int, list[tuple[int, int]]]:
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    spokes = [(i, 5 + i) for i in range(5)]
    return 10, outer + inner + spokes


GRAPHS = {"cycle": cycle, "path": path, "complete": complete,
          "grid": grid, "petersen": petersen}


def parse_graph(spec: str) -> tuple[int, list[tuple[int, int]]]:
    """
    Parse spec strings like 'K5', 'C8', 'P10', 'grid:4x5', 'petersen',
    or  'name:arg1,arg2'  e.g.  'complete:5', 'cycle:8'.
    """
    s = spec.strip()
    if s == "petersen":
        return petersen()
    # short forms
    if len(s) >= 2 and s[0] in "KCP" and s[1:].isdigit():
        return {"K": complete, "C": cycle, "P": path}[s[0]](int(s[1:]))
    if ":" in s:
        name, arg = s.split(":", 1)
        if name == "grid":
            r, c = arg.lower().split("x")
            return grid(int(r), int(c))
        if name in GRAPHS:
            return GRAPHS[name](*(int(a) for a in arg.split(",")))
    raise ValueError(f"Cannot parse graph spec: {spec!r}")


# ===================================================================
# 5.  CLI
# ===================================================================

def _print_row(label: str, z_hat: complex, z_true: complex | None,
               bound: float) -> None:
    z_str = f"Z̃ = {z_hat.real:.10f}"
    if abs(z_hat.imag) > 1e-9:
        z_str += f" + {z_hat.imag:.2e}i"
    if z_true is not None:
        rel = abs(z_hat - z_true) / abs(z_true)
        print(f"  {label:8s}  {z_str}   rel.err = {rel:.2e}   "
              f"log-err bound ≤ {bound:.2e}")
    else:
        print(f"  {label:8s}  {z_str}   log-err bound ≤ {bound:.2e}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="LSS Taylor approximation of the Ising partition function.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Graph specs:  K5, C8, P10, grid:4x5, petersen, "
            "complete:6, cycle:12, ...\n"
            "Examples:\n"
            "  python taylor_lss.py --graph K5 --beta 0.5 --lam 0.4 --m 6\n"
            "  python taylor_lss.py --graph C8 --beta 0.7 --h 0.2 --m 10 "
            "--method insects\n"
        ),
    )
    p.add_argument("--graph", required=True, help="graph spec, e.g. K5, C8, grid:3x3")
    p.add_argument("--beta", type=float, default=0.5,
                   help="edge activity β ∈ [0, 1] for ferromagnetic Ising")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--lam", type=float, default=None,
                   help="vertex activity λ (|λ| ≠ 1 required by Lee-Yang)")
    g.add_argument("--h", type=float, default=None,
                   help="external field h; λ = exp(2h)")
    p.add_argument("--m", type=int, required=True, help="Taylor truncation order")
    p.add_argument("--method", choices=["naive", "insects", "both"],
                   default="both")
    p.add_argument("--no-exact", action="store_true",
                   help="skip the exact-Z comparison (use for n > ~22)")
    args = p.parse_args()

    n, edges = parse_graph(args.graph)
    if args.lam is not None:
        lam = args.lam
    elif args.h is not None:
        lam = math.exp(2 * args.h)
    else:
        lam = 0.5  # default
    if abs(lam) == 1:
        raise SystemExit("|λ| = 1 lies on the Lee-Yang circle (excluded).")
    if abs(lam) > 1:
        print(f"note: |λ| = {abs(lam):.4f} > 1; using the λ ↔ 1/λ symmetry "
              "Z(λ) = λ^n Z(1/λ) (paper, after eq. 3).")

    print(f"Graph: {args.graph}  (n = {n}, |E| = {len(edges)})")
    print(f"β = {args.beta},  λ = {lam},  m = {args.m}")
    print()

    z_true: complex | None = None
    if not args.no_exact and n <= 22:
        coeffs_exact = z_polynomial_naive(n, edges, complex(args.beta))
        z_true = evaluate_polynomial(coeffs_exact, complex(lam))
        print(f"  exact     Z = {z_true.real:.10f}")

    bound = truncation_error_bound(n, complex(lam), args.m)

    if args.method in ("naive", "both"):
        z_hat = z_approximation(n, edges, complex(args.beta),
                                complex(lam), args.m, method="naive")
        _print_row("naive", z_hat, z_true, bound)

    if args.method in ("insects", "both"):
        z_hat = z_approximation(n, edges, complex(args.beta),
                                complex(lam), args.m, method="insects")
        _print_row("insects", z_hat, z_true, bound)


# ===================================================================
# 6.  Demo / sanity
# ===================================================================

def demo() -> None:
    print("LSS / Barvinok Taylor approximation of the Ising Z(λ)")
    print("=" * 60)

    cases = [
        ("triangle K_3",  complete(3)),
        ("4-cycle C_4",   cycle(4)),
        ("complete K_5",  complete(5)),
        ("6-cycle C_6",   cycle(6)),
        ("Petersen",      petersen()),
    ]
    beta = 0.5 + 0j
    lam = 0.4 + 0j

    for name, (n, edges) in cases:
        coeffs_true = z_polynomial_naive(n, edges, beta)
        z_true = evaluate_polynomial(coeffs_true, lam)
        print(f"\n[{name}]  n={n}, |E|={len(edges)}, β={beta.real}, λ={lam.real}")
        print(f"  Z(λ) exact = {z_true.real:.10f}")

        for m in (1, 2, 4, 8):
            z_n = z_approximation(n, edges, beta, lam, m, method="naive")
            z_i = z_approximation(n, edges, beta, lam, m, method="insects")
            rel_n = abs(z_n - z_true) / abs(z_true)
            rel_i = abs(z_i - z_true) / abs(z_true)
            agree = abs(z_n - z_i) / max(abs(z_n), 1e-30)
            bound = truncation_error_bound(n, lam, m)
            print(f"  m={m:2d}:  naive {z_n.real:.10f} (err {rel_n:.2e})   "
                  f"insects {z_i.real:.10f} (err {rel_i:.2e})   "
                  f"|n−i|/|n| = {agree:.1e}   bnd≤{bound:.1e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        demo()
