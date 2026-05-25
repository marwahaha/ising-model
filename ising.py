"""Single-site Metropolis Markov chain for the ferromagnetic Ising model.

Hamiltonian (with J = 1):
    H(sigma) = - sum_{(i,j) in E} sigma_i sigma_j  -  h * sum_i sigma_i
"""

from __future__ import annotations

import random
from typing import Dict, Hashable, Iterable, List, Sequence

Node = Hashable
Graph = Dict[Node, List[Node]]


class IsingChain:
    def __init__(self, G: Graph, h: float, beta: float, rng: random.Random | None = None):
        self.G = G
        self.nodes: List[Node] = list(G.keys())
        self.h = float(h)
        self.beta = float(beta)
        self.rng = rng if rng is not None else random.Random()

        # Initial distribution: uniform over the two ground states.
        s = 1 if self.rng.random() < 0.5 else -1
        self.sigma: Dict[Node, int] = {v: s for v in self.nodes}

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
        bond_sum = 0
        seen = set()
        for v in self.nodes:
            for u in self.G[v]:
                key = (u, v) if id(u) < id(v) else (v, u)
                if key in seen:
                    continue
                seen.add(key)
                bond_sum += self.sigma[u] * self.sigma[v]
        return -bond_sum - self.h * sum(self.sigma.values())

    def magnetization(self) -> float:
        return sum(self.sigma.values()) / len(self.nodes)

    # ----- the Markov chain -----

    def step(self) -> bool:
        """One single-site Metropolis update. Returns True if the proposal was accepted."""
        v = self.rng.choice(self.nodes)
        dE = self.delta_E_flip(v)
        if dE <= 0 or self.rng.random() < pow(2.718281828459045, -self.beta * dE):
            self.sigma[v] = -self.sigma[v]
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
