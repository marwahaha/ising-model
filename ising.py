"""Single-site Markov chains for the ferromagnetic Ising model.

Hamiltonian (with J = 1):
    H(sigma) = - sum_{(i,j) in E} sigma_i sigma_j  -  h * sum_i sigma_i

IsingChain takes a graph (adjacency dict), field strength h, inverse
temperature beta, an initial distribution, and a dynamics rule.

Initial distributions ("init" argument):
  - "ground"  : uniform over the two all-aligned ground states
  - "uniform" : uniform over all 2^|V| spin configurations

Dynamics rules ("dynamics" argument):
  - "metropolis" : pick a uniform random site v, propose flipping sigma_v;
                   accept with probability min(1, exp(-beta * dE)).  Every
                   favorable flip (dE <= 0) is accepted with probability 1.
  - "glauber"    : pick a uniform random site v, resample sigma_v from its
                   conditional distribution given neighbours.  Concretely
                   sigma_v <- +1 with probability sigmoid(2 * beta * h_eff)
                   where h_eff = (sum of neighbour spins) + h.  No proposal /
                   rejection -- the new value is drawn independent of the
                   previous one (conditional on neighbours).

Both dynamics are reversible w.r.t. the Gibbs measure and have the same
stationary distribution.  step() performs one update under the chosen
dynamics; run(n) calls step() n times.  energy(), magnetization(),
local_field(v) and delta_E_flip(v) are observables that do not mutate state.
"""

from __future__ import annotations

import random
from typing import Dict, Hashable, Iterable, List, Sequence

Node = Hashable
Graph = Dict[Node, List[Node]]


class IsingChain:
    def __init__(self, G: Graph, h: float, beta: float,
                 rng: random.Random | None = None,
                 init: str = "ground",
                 dynamics: str = "metropolis"):
        """Build the chain on graph G at field h, inverse temperature beta.

        init: "ground" (uniform over the two all-aligned ground states) or
              "uniform" (each spin iid +-1 with prob 1/2).
        dynamics: "metropolis" (propose-and-accept single-site flip) or
                  "glauber" (resample single site from its conditional given
                  neighbours).  See module docstring for details.
        """
        self.G = G
        self.nodes: List[Node] = list(G.keys())
        self.h = float(h)
        self.beta = float(beta)
        self.rng = rng if rng is not None else random.Random()

        if init == "ground":
            s = 1 if self.rng.random() < 0.5 else -1
            self.sigma: Dict[Node, int] = {v: s for v in self.nodes}
        elif init == "uniform":
            self.sigma = {v: (1 if self.rng.random() < 0.5 else -1) for v in self.nodes}
        else:
            raise ValueError(f"unknown init {init!r}; expected 'ground' or 'uniform'")

        if dynamics not in ("metropolis", "glauber"):
            raise ValueError(f"unknown dynamics {dynamics!r}; expected 'metropolis' or 'glauber'")
        self.dynamics = dynamics

        # Precompute an undirected edge list (each edge once).  Dedup by value
        # via a frozenset key -- id-based dedup is unsafe when nodes are
        # mutable objects (tuples in adjacency lists are recreated copies).
        seen = set()
        self._edges: List[Tuple[Node, Node]] = []
        for v in self.nodes:
            for u in self.G[v]:
                key = frozenset((u, v))
                if key in seen:
                    continue
                seen.add(key)
                self._edges.append((v, u))

    # ----- energy helpers -----

    def local_field(self, v: Node) -> int:
        """Sum of neighbor spins at v."""
        return sum(self.sigma[u] for u in self.G[v])

    def delta_E_flip(self, v: Node) -> float:
        """Energy change if we flip spin at v."""
        # H has two terms involving sigma_v:
        #   -sigma_v * (sum of neighbor spins)   and   -h * sigma_v
        # Flipping sigma_v -> -sigma_v changes each by a factor of -2 sigma_v.
        return 2.0 * self.sigma[v] * (self.local_field(v) + self.h)

    def energy(self) -> float:
        """Total energy of the current configuration."""
        bond_sum = sum(self.sigma[u] * self.sigma[v] for u, v in self._edges)
        return -bond_sum - self.h * sum(self.sigma.values())

    def magnetization(self) -> float:
        return sum(self.sigma.values()) / len(self.nodes)

    # ----- the Markov chain -----

    def step(self) -> bool:
        """One single-site update under the configured dynamics.  Returns True
        if the spin at the selected site actually changed value."""
        v = self.rng.choice(self.nodes)
        if self.dynamics == "metropolis":
            dE = self.delta_E_flip(v)
            if dE <= 0 or self.rng.random() < pow(2.718281828459045, -self.beta * dE):
                self.sigma[v] = -self.sigma[v]
                return True
            return False
        # Glauber / heat-bath: resample sigma_v from its conditional.
        h_eff = self.local_field(v) + self.h
        # sigmoid(2 beta h_eff) without overflow:
        z = 2.0 * self.beta * h_eff
        p_plus = 1.0 / (1.0 + pow(2.718281828459045, -z))
        new = 1 if self.rng.random() < p_plus else -1
        if new != self.sigma[v]:
            self.sigma[v] = new
            return True
        return False

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()


# --------- small graph builders for convenience ---------

def grid_2d(L: int, periodic: bool = True) -> Graph:
    """L x L square lattice as an adjacency dict with nodes (i, j)."""
    G: Graph = {(i, j): [] for i in range(L) for j in range(L)}
    for i in range(L):
        for j in range(L):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if periodic:
                    ni %= L
                    nj %= L
                elif not (0 <= ni < L and 0 <= nj < L):
                    continue
                G[(i, j)].append((ni, nj))
    return G


def complete_graph(n: int) -> Graph:
    return {i: [j for j in range(n) if j != i] for i in range(n)}


if __name__ == "__main__":
    # quick sanity demo: 20x20 lattice, slightly below 2D critical beta (~0.4407)
    G = grid_2d(20)
    chain = IsingChain(G, h=0.0, beta=0.5, rng=random.Random(0))
    print(f"start: m = {chain.magnetization():+.3f}, E = {chain.energy():.1f}")
    chain.run(50_000)
    print(f"after 50k steps: m = {chain.magnetization():+.3f}, E = {chain.energy():.1f}")
