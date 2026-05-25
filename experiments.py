"""Convergence experiments for the single-site Metropolis Ising chain on
random 3-regular graphs.

Two experiments are run, both starting from the uniform-over-two-ground-states
initial distribution (all spins +1 or all spins -1 with equal probability):

1.  n = 16 vertices.  Because 2^16 = 65536 configurations is tractable, we
    brute-force the exact equilibrium energy <E>_beta,h and plot the
    *relative error* of the running-average energy estimator,
        |E_hat_t - <E>| / |<E>|,
    as a function of Metropolis steps (log-log axes).  One figure per graph,
    one subplot per value of h, all betas overlaid as separate curves.

2.  n = 40 vertices.  2^40 is too large for brute force, so we plot the raw
    running-average energy as a function of Metropolis steps (log x-axis,
    linear y-axis) instead.  One figure per graph, one subplot per value of h,
    all betas overlaid.

In both experiments:
  - 3 independent random 3-regular graphs (deterministic seeds)
  - betas = (0.1, 0.3, 0.5, 0.8, 1, 1.2, 1.5, 2, 2.5, 5)
  - h values = (0, 0.1, 0.2, 0.5, 1)
  - 400,000 Metropolis steps per (graph, h, beta) combination

Output files: convergence_n16_graph{0,1,2}.png and convergence_n40_graph{0,1,2}.png.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ising import IsingChain, Graph


BETAS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 5.0]
H_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0]


# ----- exact equilibrium energy by brute force enumeration -----

def exact_expected_energy(edges: List[Tuple[int, int]], n: int,
                          beta: float, h: float = 0.0) -> float:
    """Return <H>_{beta,h} by summing over all 2^n spin configurations.

    Builds a (2^n, n) numpy array of spins in {-1, +1}, computes the energy
    of every configuration, and uses log-sum-exp to evaluate the Boltzmann
    expectation in a numerically stable way.  Only tractable for n <= ~24.
    """
    idx = np.arange(1 << n, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(n)[None, :]) & 1
    spins = 2 * bits - 1  # 0 -> -1, 1 -> +1
    i_arr = np.array([i for i, _ in edges])
    j_arr = np.array([j for _, j in edges])
    bond_sum = (spins[:, i_arr] * spins[:, j_arr]).sum(axis=1)
    E_arr = -bond_sum.astype(np.float64) - h * spins.sum(axis=1)
    log_w = -beta * E_arr
    log_w -= log_w.max()
    w = np.exp(log_w)
    return float((E_arr * w).sum() / w.sum())


# ----- graph adapter -----

def nx_to_adj(G_nx: nx.Graph) -> Graph:
    return {v: list(G_nx.neighbors(v)) for v in G_nx.nodes()}


# ----- one experiment: running energy for several betas on one graph -----

def run_one_graph(
    G_nx: nx.Graph,
    betas: List[float],
    h: float,
    n_steps: int,
    record_every: int,
    seed: int,
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """Run the single-site Metropolis chain on G_nx, once per beta, starting
    from the uniform-over-two-ground-states initial distribution.

    Returns a dict beta -> (steps_axis, running_mean_energy), where the
    running mean is the cumulative average of the chain's energy across all
    steps so far, recorded every `record_every` steps.
    """
    G = nx_to_adj(G_nx)
    out: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for k, beta in enumerate(betas):
        rng = random.Random(seed + k)
        chain = IsingChain(G, h=h, beta=beta, rng=rng)
        E = chain.energy()
        running_sum = 0.0
        steps_axis: List[int] = []
        running_mean: List[float] = []
        for t in range(1, n_steps + 1):
            v = chain.rng.choice(chain.nodes)
            dE = chain.delta_E_flip(v)
            if dE <= 0 or chain.rng.random() < math.exp(-chain.beta * dE):
                chain.sigma[v] = -chain.sigma[v]
                E += dE
            running_sum += E
            if t % record_every == 0:
                steps_axis.append(t)
                running_mean.append(running_sum / t)
        out[beta] = (np.array(steps_axis), np.array(running_mean))
    return out


# ----- plotting -----

def _subplot_grid(n_panels: int, ncols: int = 3) -> Tuple[plt.Figure, np.ndarray, int]:
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows),
                             squeeze=False, sharex=True)
    return fig, axes, nrows


def plot_n16_relative_error(G_nx: nx.Graph, g_idx: int, n: int,
                            betas: List[float], h_values: List[float],
                            n_steps: int, record_every: int) -> str:
    """Compute exact <E> by enumeration and plot the relative error of the
    Metropolis running-average energy as a function of step count (log-log).
    One subplot per h, all betas as overlaid curves.  Writes a PNG and
    returns its path.
    """
    edges = list(G_nx.edges())
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(betas)))
    ncols = 3
    fig, axes, _ = _subplot_grid(len(h_values), ncols=ncols)

    for h_idx, h in enumerate(h_values):
        ax = axes[h_idx // ncols][h_idx % ncols]
        print(f"  [n=16] graph {g_idx}  h = {h}")
        exact = {beta: exact_expected_energy(edges, n, beta, h=h) for beta in betas}
        traces = run_one_graph(G_nx, betas, h, n_steps, record_every,
                               seed=1000 * g_idx + 100 * h_idx)
        for beta, color in zip(betas, colors):
            steps, mean_E = traces[beta]
            denom = abs(exact[beta]) if abs(exact[beta]) > 1e-12 else 1.0
            rel_err = np.abs(mean_E - exact[beta]) / denom
            rel_err = np.maximum(rel_err, 1e-10)  # avoid log(0)
            ax.plot(steps, rel_err, color=color, linewidth=1.1, label=f"beta = {beta}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Metropolis steps")
        ax.set_ylabel(r"$|\hat E_t - \langle E \rangle| \,/\, |\langle E \rangle|$")
        ax.set_title(f"h = {h}")
        ax.grid(True, which="both", alpha=0.3)

    for k in range(len(h_values), axes.shape[0] * axes.shape[1]):
        axes[k // ncols][k % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.99, 0.02),
               fontsize=8, ncol=2, title="beta")
    fig.suptitle(f"Graph {g_idx}: 3-regular, n=16 — "
                 r"relative error of running-average energy vs. exact $\langle E \rangle$",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = f"convergence_n16_graph{g_idx}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_n40_energy(G_nx: nx.Graph, g_idx: int,
                    betas: List[float], h_values: List[float],
                    n_steps: int, record_every: int) -> str:
    """Plot the raw running-average energy of the Metropolis chain on a
    larger 3-regular graph (n = 40), where brute-force <E> is infeasible.
    One subplot per h, all betas overlaid.  Writes a PNG and returns its path.
    """
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(betas)))
    ncols = 3
    fig, axes, _ = _subplot_grid(len(h_values), ncols=ncols)

    for h_idx, h in enumerate(h_values):
        ax = axes[h_idx // ncols][h_idx % ncols]
        print(f"  [n=40] graph {g_idx}  h = {h}")
        traces = run_one_graph(G_nx, betas, h, n_steps, record_every,
                               seed=1000 * g_idx + 100 * h_idx + 7)
        for beta, color in zip(betas, colors):
            steps, mean_E = traces[beta]
            ax.plot(steps, mean_E, color=color, linewidth=1.1, label=f"beta = {beta}")
        ax.set_xscale("log")
        ax.set_xlabel("Metropolis steps")
        ax.set_ylabel(r"running average of energy  $\hat E_t$")
        ax.set_title(f"h = {h}")
        ax.grid(True, alpha=0.3)

    for k in range(len(h_values), axes.shape[0] * axes.shape[1]):
        axes[k // ncols][k % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.99, 0.02),
               fontsize=8, ncol=2, title="beta")
    fig.suptitle(f"Graph {g_idx}: 3-regular, n=40 — running-average energy "
                 r"$\hat E_t$ (no exact reference)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = f"convergence_n40_graph{g_idx}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ----- driver -----

def main():
    num_graphs = 3
    n_steps = 400_000
    record_every = 400

    # n = 16 experiments
    print("\n### n = 16 (with exact reference) ###")
    for g_idx in range(num_graphs):
        G_nx = nx.random_regular_graph(d=3, n=16, seed=100 + g_idx)
        out = plot_n16_relative_error(G_nx, g_idx, 16, BETAS, H_VALUES,
                                      n_steps, record_every)
        print(f"  wrote {out}")

    # n = 40 experiments
    print("\n### n = 40 (no exact reference) ###")
    for g_idx in range(num_graphs):
        G_nx = nx.random_regular_graph(d=3, n=40, seed=200 + g_idx)
        out = plot_n40_energy(G_nx, g_idx, BETAS, H_VALUES,
                              n_steps, record_every)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
