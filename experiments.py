"""Convergence experiments for the single-site Ising chain on random
3-regular graphs.  Compares two initial distributions x two dynamics rules.

Initial distributions ("init"):
  - "ground"  : uniform over the two all-aligned ground states (all +1 / all -1)
  - "uniform" : uniform over all 2^|V| spin configurations (each spin iid +-1)

Dynamics rules ("dyn"):
  - "metropolis" : propose flipping sigma_v; accept w.p. min(1, exp(-beta dE))
                   (every favorable flip is accepted with probability 1)
  - "glauber"    : resample sigma_v from its conditional given neighbours,
                   i.e. sigma_v <- +1 w.p. sigmoid(2 beta * (neighbours + h))

Both dynamics have the same stationary distribution but different mixing
properties.

Two experiment sizes are run:

1.  n = 16 vertices.  Because 2^16 = 65536 configurations is tractable, we
    brute-force the exact equilibrium energy <E>_{beta,h} and plot the
    *relative error* of the running-average energy estimator,
        |E_hat_t - <E>| / |<E>|,
    versus Metropolis-or-Glauber step count (log-log axes).

2.  n = 40 vertices.  2^40 is too large for brute force, so we plot the raw
    running-average energy versus step count (log x, linear y).

For each (n, graph) the simulation traces are computed once for every
(h, beta, init, dyn) combination, and rendered in two ways:
  - a static matplotlib PNG (convergence_n{N}_graph{G}.png) -- 40 curves
    per subplot, line style encodes (init, dyn).  Dense but archive-friendly.
  - a standalone plotly HTML page (convergence_n{N}_graph{G}.html) -- same
    data, but legend toggling makes it usable: each legend entry corresponds
    to a unique (beta, init, dyn) triple and toggles its curves across all
    h subplots simultaneously.  Hover shows (step, value, beta, init, dyn, h).

Line-style encoding for (init, dyn) in both renderers:
    ground   + metropolis  -> solid
    uniform  + metropolis  -> long dash
    ground   + glauber     -> dotted
    uniform  + glauber     -> dash-dot

Settings:
  - 3 independent random 3-regular graphs (deterministic seeds)
  - betas = (0.1, 0.3, 0.5, 0.8, 1, 1.2, 1.5, 2, 2.5, 5)
  - h values = (0, 0.1, 0.2, 0.5, 1)
  - 400,000 update steps per (graph, h, beta, init, dyn)
  - Traces are sub-sampled at ~120 log-spaced step counts for plotting.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ising import IsingChain, Graph


BETAS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 5.0]
H_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0]
INITS = ["ground", "uniform"]
DYNAMICS = ["metropolis", "glauber"]
NUM_LOG_SAMPLES = 120

# (init, dynamics) -> matplotlib linestyle and plotly dash
STYLE_MPL = {
    ("ground", "metropolis"): "-",
    ("uniform", "metropolis"): (0, (6, 2)),
    ("ground", "glauber"): ":",
    ("uniform", "glauber"): "-.",
}
STYLE_PLOTLY = {
    ("ground", "metropolis"): "solid",
    ("uniform", "metropolis"): "dash",
    ("ground", "glauber"): "dot",
    ("uniform", "glauber"): "dashdot",
}

# data[h][beta][(init, dyn)] = (steps, running_mean_energy)
TraceData = Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]]


# ----- exact equilibrium energy by brute force enumeration -----

def exact_expected_energy(edges: List[Tuple[int, int]], n: int,
                          beta: float, h: float = 0.0) -> float:
    """Return <H>_{beta,h} by summing over all 2^n spin configurations.

    Builds a (2^n, n) numpy array of spins in {-1, +1}, computes the energy of
    every configuration, and uses log-sum-exp for a numerically stable
    Boltzmann expectation.  Only tractable for n <= ~24.
    """
    idx = np.arange(1 << n, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(n)[None, :]) & 1
    spins = 2 * bits - 1
    i_arr = np.array([i for i, _ in edges])
    j_arr = np.array([j for _, j in edges])
    bond_sum = (spins[:, i_arr] * spins[:, j_arr]).sum(axis=1)
    E_arr = -bond_sum.astype(np.float64) - h * spins.sum(axis=1)
    log_w = -beta * E_arr
    log_w -= log_w.max()
    w = np.exp(log_w)
    return float((E_arr * w).sum() / w.sum())


def nx_to_adj(G_nx: nx.Graph) -> Graph:
    return {v: list(G_nx.neighbors(v)) for v in G_nx.nodes()}


# ----- Markov chain runs -----

def _log_record_steps(n_steps: int, n_points: int) -> np.ndarray:
    return np.unique(np.round(np.geomspace(1, n_steps, num=n_points)).astype(int))


def run_one_graph(
    G_nx: nx.Graph,
    betas: List[float],
    h: float,
    init: str,
    dynamics: str,
    n_steps: int,
    record_at: np.ndarray,
    seed: int,
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """Run the chosen single-site dynamics on G_nx, once per beta, starting
    from the named initial distribution.  Returns, per beta, the running-
    average energy sampled at the step counts in record_at.

    The inner loop is inlined (rather than calling chain.step()) so we can
    incrementally update the energy without recomputing it from scratch.
    """
    G = nx_to_adj(G_nx)
    record_set = set(int(s) for s in record_at)
    e = math.e
    out: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for k, beta in enumerate(betas):
        rng = random.Random(seed + k)
        chain = IsingChain(G, h=h, beta=beta, rng=rng, init=init, dynamics=dynamics)
        E = chain.energy()
        running_sum = 0.0
        steps_axis: List[int] = []
        running_mean: List[float] = []
        nodes = chain.nodes
        sigma = chain.sigma
        adj = chain.G
        for t in range(1, n_steps + 1):
            v = rng.choice(nodes)
            if dynamics == "metropolis":
                # dE = 2 * sigma_v * (sum_neighbours + h)
                neighbour_sum = sum(sigma[u] for u in adj[v])
                dE = 2.0 * sigma[v] * (neighbour_sum + h)
                if dE <= 0 or rng.random() < pow(e, -beta * dE):
                    sigma[v] = -sigma[v]
                    E += dE
            else:  # glauber
                neighbour_sum = sum(sigma[u] for u in adj[v])
                h_eff = neighbour_sum + h
                z = 2.0 * beta * h_eff
                # sigmoid(z), stable
                if z >= 0:
                    p_plus = 1.0 / (1.0 + pow(e, -z))
                else:
                    ez = pow(e, z)
                    p_plus = ez / (1.0 + ez)
                new = 1 if rng.random() < p_plus else -1
                if new != sigma[v]:
                    dE = 2.0 * sigma[v] * (neighbour_sum + h)
                    sigma[v] = new
                    E += dE
            running_sum += E
            if t in record_set:
                steps_axis.append(t)
                running_mean.append(running_sum / t)
        out[beta] = (np.array(steps_axis), np.array(running_mean))
    return out


def compute_all_traces(G_nx: nx.Graph, g_idx: int, n_label: str,
                       h_values: List[float], betas: List[float],
                       inits: List[str], dynamics_list: List[str],
                       n_steps: int, base_seed: int) -> TraceData:
    """Run the chain for every (h, beta, init, dyn) combination on G_nx and
    return a nested dict data[h][beta][(init, dyn)] = (steps, running_mean).
    """
    record_at = _log_record_steps(n_steps, NUM_LOG_SAMPLES)
    data: TraceData = {h: {beta: {} for beta in betas} for h in h_values}
    for h_idx, h in enumerate(h_values):
        for init in inits:
            for dyn in dynamics_list:
                print(f"  [{n_label}] graph {g_idx}  h = {h}  init = {init}  dyn = {dyn}")
                seed = (base_seed
                        + 100 * h_idx
                        + (0 if init == "ground" else 50_000)
                        + (0 if dyn == "metropolis" else 1_000_000))
                traces = run_one_graph(G_nx, betas, h, init, dyn, n_steps, record_at, seed)
                for beta in betas:
                    data[h][beta][(init, dyn)] = traces[beta]
    return data


# ----- static (matplotlib) plotting -----

def _subplot_grid(n_panels: int, ncols: int = 3) -> Tuple[plt.Figure, np.ndarray]:
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.4 * nrows),
                             squeeze=False, sharex=True)
    return fig, axes


def _add_combined_legend_mpl(fig: plt.Figure, betas: List[float], colors) -> None:
    beta_handles = [plt.Line2D([], [], color=c, linewidth=1.8, label=f"beta = {b}")
                    for b, c in zip(betas, colors)]
    style_handles = [
        plt.Line2D([], [], color="black", linewidth=1.8, linestyle="-",
                   label='ground / metropolis'),
        plt.Line2D([], [], color="black", linewidth=1.8, linestyle=(0, (6, 2)),
                   label='uniform / metropolis'),
        plt.Line2D([], [], color="black", linewidth=1.8, linestyle=":",
                   label='ground / glauber'),
        plt.Line2D([], [], color="black", linewidth=1.8, linestyle="-.",
                   label='uniform / glauber'),
    ]
    leg1 = fig.legend(handles=beta_handles, loc="lower right",
                      bbox_to_anchor=(0.99, 0.04), fontsize=8, ncol=2,
                      title="beta", framealpha=0.9)
    fig.add_artist(leg1)
    fig.legend(handles=style_handles, loc="lower left",
               bbox_to_anchor=(0.01, 0.04), fontsize=8,
               title="(init, dynamics)", framealpha=0.9)


def static_plot(data: TraceData, g_idx: int, n: int,
                h_values: List[float], betas: List[float],
                inits: List[str], dynamics_list: List[str],
                exact: Optional[Dict[float, Dict[float, float]]],
                out_path: str, title_suffix: str, y_label: str) -> None:
    """Static matplotlib PNG with 40 curves per subplot.  Best viewed via the
    matching interactive HTML; this file is provided for archival.
    """
    colors = plt.cm.tab10(np.arange(len(betas)) % 10)
    ncols = 3
    fig, axes = _subplot_grid(len(h_values), ncols=ncols)
    use_rel_err = exact is not None
    for h_idx, h in enumerate(h_values):
        ax = axes[h_idx // ncols][h_idx % ncols]
        for init in inits:
            for dyn in dynamics_list:
                style = STYLE_MPL[(init, dyn)]
                for beta, color in zip(betas, colors):
                    steps, mean_E = data[h][beta][(init, dyn)]
                    if use_rel_err:
                        ref = exact[h][beta]
                        denom = abs(ref) if abs(ref) > 1e-12 else 1.0
                        y = np.maximum(np.abs(mean_E - ref) / denom, 1e-10)
                    else:
                        y = mean_E
                    ax.plot(steps, y, color=color, linewidth=1.15,
                            linestyle=style, alpha=0.85)
        ax.set_xscale("log")
        if use_rel_err:
            ax.set_yscale("log")
        ax.set_xlabel("update steps")
        ax.set_ylabel(y_label)
        ax.set_title(f"h = {h}")
        ax.grid(True, which="both", alpha=0.3)
    for k in range(len(h_values), axes.shape[0] * axes.shape[1]):
        axes[k // ncols][k % ncols].axis("off")
    _add_combined_legend_mpl(fig, betas, colors)
    fig.suptitle(f"Graph {g_idx}: 3-regular, n={n} — {title_suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----- interactive (plotly) plotting -----

PLOTLY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def interactive_plot(data: TraceData, g_idx: int, n: int,
                     h_values: List[float], betas: List[float],
                     inits: List[str], dynamics_list: List[str],
                     exact: Optional[Dict[float, Dict[float, float]]],
                     out_path: str, title_suffix: str,
                     y_label_html: str) -> None:
    """Standalone plotly HTML.  Each legend entry is a unique (beta, init,
    dyn) triple; clicking it toggles its 5 subplot traces simultaneously
    (via shared `legendgroup` and `showlegend=True` only on the first
    subplot's traces).  Double-click isolates.  Hover shows step and value.
    """
    use_rel_err = exact is not None
    ncols = 3
    nrows = math.ceil(len(h_values) / ncols)
    titles = [f"h = {h}" for h in h_values]
    titles += [""] * (nrows * ncols - len(h_values))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        shared_xaxes=True,
                        horizontal_spacing=0.07, vertical_spacing=0.12)

    for h_idx, h in enumerate(h_values):
        row, col = h_idx // ncols + 1, h_idx % ncols + 1
        is_first = (h_idx == 0)
        for b_idx, beta in enumerate(betas):
            color = PLOTLY_COLORS[b_idx % len(PLOTLY_COLORS)]
            for init in inits:
                for dyn in dynamics_list:
                    steps, mean_E = data[h][beta][(init, dyn)]
                    if use_rel_err:
                        ref = exact[h][beta]
                        denom = abs(ref) if abs(ref) > 1e-12 else 1.0
                        y = np.maximum(np.abs(mean_E - ref) / denom, 1e-10)
                    else:
                        y = mean_E
                    group = f"beta={beta} init={init} dyn={dyn}"
                    fig.add_trace(
                        go.Scatter(
                            x=steps, y=y, mode="lines",
                            name=f"β={beta}, {init}, {dyn}",
                            legendgroup=group,
                            showlegend=is_first,
                            line=dict(color=color, width=1.6,
                                      dash=STYLE_PLOTLY[(init, dyn)]),
                            hovertemplate=(
                                f"<b>β={beta}</b><br>"
                                f"init={init}<br>"
                                f"dynamics={dyn}<br>"
                                f"h={h}<br>"
                                "steps=%{x:,}<br>"
                                f"{y_label_html}=%{{y:.4g}}<extra></extra>"
                            ),
                        ),
                        row=row, col=col,
                    )

        fig.update_xaxes(type="log", title_text="update steps", row=row, col=col)
        if use_rel_err:
            fig.update_yaxes(type="log", title_text=y_label_html, row=row, col=col)
        else:
            fig.update_yaxes(title_text=y_label_html, row=row, col=col)

    fig.update_layout(
        title=dict(text=(f"<b>Graph {g_idx}</b>: 3-regular, n={n} — {title_suffix}"
                         "<br><sub>Click a legend entry to toggle a (beta, init, dyn) "
                         "curve across all panels. Double-click to isolate.</sub>"),
                   x=0.5, xanchor="center"),
        height=320 * nrows + 130,
        width=420 * ncols + 320,
        hovermode="closest",
        legend=dict(title="(beta, init, dynamics)", itemsizing="constant",
                    bgcolor="rgba(255,255,255,0.92)",
                    groupclick="toggleitem"),
        template="plotly_white",
    )
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


# ----- driver -----

def main():
    num_graphs = 3
    n_steps = 400_000

    print("\n### n = 16 (with exact reference) ###")
    for g_idx in range(num_graphs):
        G_nx = nx.random_regular_graph(d=3, n=16, seed=100 + g_idx)
        edges = list(G_nx.edges())
        exact = {
            h: {beta: exact_expected_energy(edges, 16, beta, h=h) for beta in BETAS}
            for h in H_VALUES
        }
        data = compute_all_traces(G_nx, g_idx, "n=16", H_VALUES, BETAS,
                                  INITS, DYNAMICS,
                                  n_steps=n_steps, base_seed=1000 * g_idx)
        static_plot(
            data, g_idx, n=16, h_values=H_VALUES, betas=BETAS,
            inits=INITS, dynamics_list=DYNAMICS, exact=exact,
            out_path=f"convergence_n16_graph{g_idx}.png",
            title_suffix=r"relative error of running-average energy vs. exact $\langle E \rangle$",
            y_label=r"$|\hat E_t - \langle E \rangle| \,/\, |\langle E \rangle|$",
        )
        interactive_plot(
            data, g_idx, n=16, h_values=H_VALUES, betas=BETAS,
            inits=INITS, dynamics_list=DYNAMICS, exact=exact,
            out_path=f"convergence_n16_graph{g_idx}.html",
            title_suffix="relative error of running-average energy vs. exact ⟨E⟩",
            y_label_html="|Ê - ⟨E⟩| / |⟨E⟩|",
        )
        print(f"  wrote convergence_n16_graph{g_idx}.{{png,html}}")

    print("\n### n = 40 (no exact reference) ###")
    for g_idx in range(num_graphs):
        G_nx = nx.random_regular_graph(d=3, n=40, seed=200 + g_idx)
        data = compute_all_traces(G_nx, g_idx, "n=40", H_VALUES, BETAS,
                                  INITS, DYNAMICS,
                                  n_steps=n_steps, base_seed=1000 * g_idx + 7)
        static_plot(
            data, g_idx, n=40, h_values=H_VALUES, betas=BETAS,
            inits=INITS, dynamics_list=DYNAMICS, exact=None,
            out_path=f"convergence_n40_graph{g_idx}.png",
            title_suffix=r"running-average energy $\hat E_t$ (no exact reference)",
            y_label=r"running average of energy $\hat E_t$",
        )
        interactive_plot(
            data, g_idx, n=40, h_values=H_VALUES, betas=BETAS,
            inits=INITS, dynamics_list=DYNAMICS, exact=None,
            out_path=f"convergence_n40_graph{g_idx}.html",
            title_suffix="running-average energy Ê (no exact reference)",
            y_label_html="running average of energy Ê",
        )
        print(f"  wrote convergence_n40_graph{g_idx}.{{png,html}}")


if __name__ == "__main__":
    main()
