"""Render PNGs and a single combined interactive HTML from the CSVs written
by simulate.py.

Outputs:
  - convergence_{graph_id}.png  : static matplotlib PNG per graph (archive).
        40 curves per panel (10 betas x 2 inits x 2 dynamics).
        Line-style encodes (init, dynamics):
            ground   + metropolis  -> solid
            uniform  + metropolis  -> long dash
            ground   + glauber     -> dotted
            uniform  + glauber     -> dash-dot
  - convergence.html : ONE interactive page with every graph baked in.  Top
        controls let you:
          * pick which graph to display (radio buttons; one at a time;
            default: the first graph encountered, n16_graph0)
          * show/hide ground / uniform / metropolis / glauber
          * pick which beta values to show
        For the visible graph, both an interactive node-link diagram
        (spring layout) and the 5-h convergence chart are shown.
        Each (beta, init, dyn) trace shares a `legendgroup` spanning every
        h subplot, with `legend.groupclick = "togglegroup"`, so single-
        clicking a legend entry hides the curve in every panel.
        Hover shows (step, value, beta, init, dynamics, h).

For n where exact <E> is available (n <= 20) the y axis is relative error
        |E_hat_t - <E>| / |<E>|
on log-log axes; otherwise it is the raw running-mean energy on log-x,
linear-y axes.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DATA_DIR = "data"
GRAPHS_SUBDIR = os.path.join(DATA_DIR, "graphs")
TRACES_CSV = os.path.join(DATA_DIR, "traces.csv")
EXACT_CSV = os.path.join(DATA_DIR, "exact.csv")
LOG_Z_CSV = os.path.join(DATA_DIR, "log_z.csv")
LOG_Z_BUDGET_CSV = os.path.join(DATA_DIR, "log_z_budget.csv")
LOG_Z_MCMC_CSV = os.path.join(DATA_DIR, "log_z_mcmc.csv")
COMBINED_HTML = "convergence.html"

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
PLOTLY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------- data loading ----------

def load_traces(path: str):
    """Return (data, graph_ids, betas, h_values, inits, dyns).

    data[graph_id][h][beta][(init, dyn)] = (steps_array, running_mean_array)
    `graph_ids` is sorted so that n16_* comes before n40_* and lexicographic
    within each size.
    """
    raw = defaultdict(list)
    h_set, beta_set, init_set, dyn_set = set(), set(), set(), set()
    graph_id_set = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = row["graph_id"]
            h = float(row["h"])
            beta = float(row["beta"])
            init = row["init"]
            dyn = row["dynamics"]
            step = int(row["step"])
            mean_E = float(row["running_mean_E"])
            raw[(gid, h, beta, init, dyn)].append((step, mean_E))
            graph_id_set.add(gid)
            h_set.add(h); beta_set.add(beta); init_set.add(init); dyn_set.add(dyn)

    data: Dict = {}
    for (gid, h, beta, init, dyn), pairs in raw.items():
        pairs.sort()
        steps = np.array([p[0] for p in pairs])
        mean = np.array([p[1] for p in pairs])
        data.setdefault(gid, {}).setdefault(h, {}).setdefault(beta, {})[(init, dyn)] = (steps, mean)

    def gid_key(gid):
        # n16_graphX -> (16, X), n40_graphX -> (40, X)
        n_part, g_part = gid.split("_")
        return (int(n_part[1:]), int(g_part.replace("graph", "")))

    graph_ids = sorted(graph_id_set, key=gid_key)
    return data, graph_ids, sorted(beta_set), sorted(h_set), sorted(init_set), sorted(dyn_set)


def load_exact(path: str) -> Dict[Tuple[str, float, float], float]:
    out: Dict[Tuple[str, float, float], float] = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[(row["graph_id"], float(row["h"]), float(row["beta"]))] = float(row["exact_E"])
    return out


def load_log_z(path: str) -> Dict[str, Dict[float, Dict[float, Dict[str, float]]]]:
    """log_z[graph_id][h][beta][method] = log_Z.  Empty if the file is absent."""
    out: Dict = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gid = row["graph_id"]
            h = float(row["h"])
            beta = float(row["beta"])
            method = row["method"]
            lz = float(row["log_Z"])
            out.setdefault(gid, {}).setdefault(h, {}).setdefault(beta, {})[method] = lz
    return out


def load_log_z_mcmc(path: str
                    ) -> Dict[str, Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]]]:
    """log_z_mcmc[graph_id][h][beta][(init, dyn)] = (steps_array, log_z_array).
    Derived from data/traces.csv via run_thermo_integration.py."""
    out: Dict = {}
    if not os.path.exists(path):
        return out
    raw = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gid = row["graph_id"]
            h = float(row["h"])
            beta = float(row["beta"])
            init = row["init"]
            dyn = row["dynamics"]
            step = int(row["step"])
            lz = float(row["log_z_mcmc"])
            raw[(gid, h, beta, init, dyn)].append((step, lz))
    for (gid, h, beta, init, dyn), pairs in raw.items():
        pairs.sort()
        steps = np.array([p[0] for p in pairs])
        lz = np.array([p[1] for p in pairs])
        (out.setdefault(gid, {}).setdefault(h, {})
            .setdefault(beta, {})[(init, dyn)]) = (steps, lz)
    return out


def load_log_z_budget(path: str
                      ) -> Dict[str, Dict[float, Dict[float, Dict[int, List[Dict]]]]]:
    """log_z_budget[graph_id][h][beta][step_n_mult] = [
            {m, n_segs, total_steps, log_Z}, ...  (one dict per samples-per-segment)
       ].  Sorted by total_steps ascending.  Missing/failed runs have log_Z = NaN.
    If the CSV has no `step_n_mult` column (older format) it's treated as 1."""
    out: Dict = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gid = row["graph_id"]
            h = float(row["h"])
            beta = float(row["beta"])
            step_n_mult = int(row.get("step_n_mult", 1) or 1)
            try:
                log_Z = float(row["log_Z"])
            except ValueError:
                log_Z = float("nan")
            rec = dict(
                m=int(row["samples_per_segment"]),
                n_segs=int(row["n_segments"]),
                total_steps=int(row["total_steps"]),
                log_Z=log_Z,
            )
            (out.setdefault(gid, {}).setdefault(h, {})
                .setdefault(beta, {}).setdefault(step_n_mult, []).append(rec))
    for gid in out:
        for h in out[gid]:
            for beta in out[gid][h]:
                for s in out[gid][h][beta]:
                    out[gid][h][beta][s].sort(key=lambda r: r["total_steps"])
    return out


def load_graph(graph_id: str) -> nx.Graph:
    with open(os.path.join(GRAPHS_SUBDIR, f"{graph_id}.json")) as f:
        gj = json.load(f)
    G = nx.Graph()
    G.add_nodes_from(range(gj["n"]))
    G.add_edges_from([tuple(e) for e in gj["edges"]])
    return G


# ---------- static matplotlib plot (per graph, archival) ----------

def _subplot_grid(n_panels: int, ncols: int = 3) -> Tuple[plt.Figure, np.ndarray]:
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.4 * nrows),
                             squeeze=False, sharex=True)
    return fig, axes


def static_plot(graph_id: str, n: int, graph_data: Dict,
                exact: Dict[Tuple[str, float, float], float],
                betas: List[float], h_values: List[float],
                inits: List[str], dyns: List[str],
                out_path: str) -> None:
    has_exact = any((graph_id, h, beta) in exact for h in h_values for beta in betas)
    colors = plt.cm.tab10(np.arange(len(betas)) % 10)
    ncols = 3
    fig, axes = _subplot_grid(len(h_values), ncols=ncols)
    for h_idx, h in enumerate(h_values):
        ax = axes[h_idx // ncols][h_idx % ncols]
        if h not in graph_data:
            ax.set_title(f"h = {h}  (no data)")
            continue
        for init in inits:
            for dyn in dyns:
                style = STYLE_MPL[(init, dyn)]
                for beta, color in zip(betas, colors):
                    if beta not in graph_data[h] or (init, dyn) not in graph_data[h][beta]:
                        continue
                    steps, mean_E = graph_data[h][beta][(init, dyn)]
                    if has_exact:
                        ref = exact[(graph_id, h, beta)]
                        denom = abs(ref) if abs(ref) > 1e-12 else 1.0
                        y = np.maximum(np.abs(mean_E - ref) / denom, 1e-10)
                    else:
                        y = mean_E
                    ax.plot(steps, y, color=color, linewidth=1.15,
                            linestyle=style, alpha=0.85)
        ax.set_xscale("log")
        if has_exact:
            ax.set_yscale("log")
        ax.set_xlabel("update steps")
        ax.set_ylabel(r"$|\hat E_t - \langle E \rangle| / |\langle E \rangle|$"
                      if has_exact else r"running average of energy  $\hat E_t$")
        ax.set_title(f"h = {h}")
        ax.grid(True, which="both", alpha=0.3)
    for k in range(len(h_values), axes.shape[0] * axes.shape[1]):
        axes[k // ncols][k % ncols].axis("off")

    beta_handles = [plt.Line2D([], [], color=c, linewidth=1.8, label=f"beta = {b}")
                    for b, c in zip(betas, colors)]
    style_handles = [
        plt.Line2D([], [], color="black", linewidth=1.8, linestyle="-",
                   label="ground / metropolis"),
        plt.Line2D([], [], color="black", linewidth=1.8, linestyle=(0, (6, 2)),
                   label="uniform / metropolis"),
        plt.Line2D([], [], color="black", linewidth=1.8, linestyle=":",
                   label="ground / glauber"),
        plt.Line2D([], [], color="black", linewidth=1.8, linestyle="-.",
                   label="uniform / glauber"),
    ]
    leg1 = fig.legend(handles=beta_handles, loc="lower right",
                      bbox_to_anchor=(0.99, 0.04), fontsize=8, ncol=2,
                      title="beta", framealpha=0.9)
    fig.add_artist(leg1)
    fig.legend(handles=style_handles, loc="lower left",
               bbox_to_anchor=(0.01, 0.04), fontsize=8,
               title="(init, dynamics)", framealpha=0.9)
    title_suffix = (r"relative error of running-average energy vs. exact $\langle E \rangle$"
                    if has_exact else
                    r"running-average energy $\hat E_t$ (no exact reference)")
    fig.suptitle(f"{graph_id}: 3-regular, n={n} — {title_suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------- plotly figure builders ----------

def _graph_figure(G_nx: nx.Graph, title: str) -> go.Figure:
    pos = nx.spring_layout(G_nx, seed=0)
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    for u, v in G_nx.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    nodes_x = [pos[v][0] for v in G_nx.nodes()]
    nodes_y = [pos[v][1] for v in G_nx.nodes()]
    node_labels = [str(v) for v in G_nx.nodes()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.8, color="#888"),
                             hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter(x=nodes_x, y=nodes_y, mode="markers+text",
                             marker=dict(size=18, color="#1f77b4",
                                         line=dict(color="white", width=1.2)),
                             text=node_labels, textposition="middle center",
                             textfont=dict(size=10, color="white"),
                             hovertemplate="vertex %{text}<extra></extra>",
                             showlegend=False))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=13)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=30, b=10),
        height=320,
        dragmode="pan",
        template="plotly_white",
    )
    return fig


def _convergence_figure(graph_id: str, graph_data: Dict,
                        exact: Dict[Tuple[str, float, float], float],
                        betas: List[float], h_values: List[float],
                        inits: List[str], dyns: List[str]) -> Tuple[go.Figure, bool]:
    has_exact = any((graph_id, h, beta) in exact for h in h_values for beta in betas)
    ncols = 3
    nrows = math.ceil(len(h_values) / ncols)
    titles = [f"h = {h}" for h in h_values] + [""] * (nrows * ncols - len(h_values))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        shared_xaxes=True,
                        horizontal_spacing=0.07, vertical_spacing=0.12)
    for h_idx, h in enumerate(h_values):
        row, col = h_idx // ncols + 1, h_idx % ncols + 1
        is_first = (h_idx == 0)
        if h not in graph_data:
            continue
        for b_idx, beta in enumerate(betas):
            color = PLOTLY_COLORS[b_idx % len(PLOTLY_COLORS)]
            for init in inits:
                for dyn in dyns:
                    if beta not in graph_data[h] or (init, dyn) not in graph_data[h][beta]:
                        continue
                    steps, mean_E = graph_data[h][beta][(init, dyn)]
                    if has_exact:
                        ref = exact[(graph_id, h, beta)]
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
                                "y=%{y:.4g}<extra></extra>"
                            ),
                        ),
                        row=row, col=col,
                    )
        fig.update_xaxes(type="log", title_text="update steps", row=row, col=col)
        if has_exact:
            fig.update_yaxes(type="log",
                             title_text="|Ê - ⟨E⟩| / |⟨E⟩|", row=row, col=col)
        else:
            fig.update_yaxes(title_text="running average of energy Ê", row=row, col=col)
    fig.update_layout(
        title=dict(text=("Relative error of running-average energy vs. exact ⟨E⟩"
                         if has_exact else
                         "Running-average energy Ê (no exact reference for n>20)"),
                   x=0.5, xanchor="center", font=dict(size=13)),
        height=320 * nrows + 120,
        hovermode="closest",
        dragmode="pan",
        legend=dict(title="(beta, init, dynamics)", itemsizing="constant",
                    bgcolor="rgba(255,255,255,0.92)",
                    groupclick="togglegroup"),
        template="plotly_white",
    )
    return fig, has_exact


# ---------- log Z convergence figure ----------

STEP_N_MULT_DASH = {1: "solid", 2: "dash", 4: "dash", 5: "dot", 20: "dot"}
STEP_N_MULT_WIDTH = {1: 1.6, 2: 1.2, 4: 1.4, 5: 1.2, 20: 1.4}
STEP_N_MULT_MARKER = {1: "circle", 2: "square", 4: "square", 5: "diamond", 20: "diamond"}


def _combined_log_z_figure(graph_id: str, n: int,
                           log_z_mcmc_for_graph: Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]],
                           log_z_budget_for_graph: Dict[float, Dict[float, Dict[int, List[Dict]]]],
                           log_z_final_for_graph: Dict[float, Dict[float, Dict[str, float]]],
                           betas: List[float], h_values: List[float],
                           inits: List[str], dyns: List[str]
                           ) -> Tuple[go.Figure, bool]:
    """Two sub-panels per h, stacked: MCMC + thermo integration on top,
    JS FPRAS budget sweep below.  Both use x = total chain steps so the
    panels are directly comparable.
      - MCMC thermo at recorded step t for target beta_i:
            x = (i+1) * t   (i+1 spin chains contribute to the integral)
      - FPRAS at (beta, m, step_n_mult): x = total_steps from the CSV.
    y is relative error vs exact log Z when n <= 20, raw log Z otherwise."""
    has_exact = n <= 20
    nrows = 2 * len(h_values)
    titles: List[str] = []
    for h in h_values:
        titles.append(f"h = {h}  ·  MCMC + thermodynamic integration")
        titles.append(f"h = {h}  ·  JS FPRAS budget sweep")
    fig = make_subplots(rows=nrows, cols=1, subplot_titles=titles,
                        shared_xaxes=False,
                        vertical_spacing=0.04)
    for h_idx, h in enumerate(h_values):
        mcmc_row = 2 * h_idx + 1
        fpras_row = 2 * h_idx + 2
        is_first = (h_idx == 0)
        h_mcmc = log_z_mcmc_for_graph.get(h, {})
        h_budget = log_z_budget_for_graph.get(h, {})
        h_finals = log_z_final_for_graph.get(h, {})
        for b_idx, beta in enumerate(betas):
            color = PLOTLY_COLORS[b_idx % len(PLOTLY_COLORS)]
            if has_exact:
                exact_lz = h_finals.get(beta, {}).get("exact")
                if exact_lz is None:
                    continue
                denom = abs(exact_lz) if abs(exact_lz) > 1e-12 else 1.0
            n_chains_for_target = b_idx + 1

            # MCMC thermo on top sub-panel.
            beta_mcmc = h_mcmc.get(beta, {})
            for init in inits:
                for dyn in dyns:
                    if (init, dyn) not in beta_mcmc:
                        continue
                    steps, lz = beta_mcmc[(init, dyn)]
                    if has_exact:
                        y = np.maximum(np.abs(lz - exact_lz) / denom, 1e-10)
                    else:
                        y = lz
                    x = steps * n_chains_for_target
                    group = f"beta={beta} init={init} dyn={dyn}"
                    fig.add_trace(
                        go.Scatter(
                            x=x, y=y, mode="lines",
                            name=f"β={beta}, {init}, {dyn}",
                            legendgroup=group, showlegend=is_first,
                            line=dict(color=color, width=1.4,
                                      dash=STYLE_PLOTLY[(init, dyn)]),
                            hovertemplate=(
                                f"<b>β={beta}, h={h}</b><br>"
                                f"init={init}, dynamics={dyn}<br>"
                                "total chain steps=%{x:,}<br>"
                                "y=%{y:.4g}<extra></extra>"
                            ),
                        ),
                        row=mcmc_row, col=1,
                    )

            # FPRAS budget on bottom sub-panel.
            beta_budget = h_budget.get(beta, {})
            for step_n_mult in sorted(beta_budget.keys()):
                recs = beta_budget[step_n_mult]
                xs, ys, hovers = [], [], []
                for r in recs:
                    lz = r["log_Z"]
                    if math.isnan(lz):
                        continue
                    if has_exact:
                        yv = max(abs(lz - exact_lz) / denom, 1e-10)
                    else:
                        yv = lz
                    xs.append(r["total_steps"])
                    ys.append(yv)
                    hovers.append(
                        f"<b>β={beta}, h={h}</b><br>"
                        f"step = 1/({step_n_mult}n)<br>"
                        f"samples/segment = {r['m']:,}<br>"
                        f"n_segments = {r['n_segs']}<br>"
                        f"total steps = {r['total_steps']:,}<br>"
                        f"log Ẑ = {lz:.4f}"
                    )
                if not xs:
                    continue
                symbol = STEP_N_MULT_MARKER.get(step_n_mult, "circle")
                dash = STEP_N_MULT_DASH.get(step_n_mult, "solid")
                group = f"beta={beta} step={step_n_mult}"
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines+markers",
                        name=f"β={beta}" + ("" if step_n_mult == 1
                                            else f"  step 1/{step_n_mult}n"),
                        legendgroup=group, showlegend=False,
                        line=dict(color=color, width=1.0, dash=dash),
                        marker=dict(color=color, size=8, symbol=symbol,
                                    line=dict(color="white", width=1)),
                        hovertext=hovers, hoverinfo="text",
                    ),
                    row=fpras_row, col=1,
                )
        fig.update_xaxes(type="log", title_text="total chain steps",
                         row=mcmc_row, col=1)
        fig.update_xaxes(type="log", title_text="total chain steps",
                         row=fpras_row, col=1)
        if has_exact:
            fig.update_yaxes(type="log",
                             title_text="|log Ẑ − log Z| / |log Z|",
                             row=mcmc_row, col=1)
            fig.update_yaxes(type="log",
                             title_text="|log Ẑ − log Z| / |log Z|",
                             row=fpras_row, col=1)
        else:
            fig.update_yaxes(title_text="log Ẑ", row=mcmc_row, col=1)
            fig.update_yaxes(title_text="log Ẑ", row=fpras_row, col=1)
    fig.update_layout(
        title=dict(text="log Z: MCMC+thermo (top) vs JS FPRAS budget (bottom), per h",
                   x=0.5, xanchor="center", font=dict(size=12)),
        height=260 * nrows + 100,
        hovermode="closest",
        dragmode="pan",
        legend=dict(title="(β, init, dyn)  — MCMC only",
                    itemsizing="constant",
                    bgcolor="rgba(255,255,255,0.92)",
                    groupclick="togglegroup"),
        template="plotly_white",
    )
    return fig, has_exact


def _log_z_mcmc_figure(graph_id: str,
                       log_z_mcmc_for_graph: Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]],
                       log_z_final_for_graph: Dict[float, Dict[float, Dict[str, float]]],
                       betas: List[float], h_values: List[float],
                       inits: List[str], dyns: List[str]
                       ) -> Tuple[go.Figure, bool]:
    """log Z derived from the spin chains' <E>(beta, step) traces via
    trapezoidal thermodynamic integration over beta.  Same 4 (init, dyn) x
    10 beta structure as the energy view; y axis is relative error vs exact
    log Z when n <= 20."""
    has_exact = any("exact" in log_z_final_for_graph.get(h, {}).get(beta, {})
                    for h in h_values for beta in betas)
    ncols = 3
    nrows = math.ceil(len(h_values) / ncols)
    titles = [f"h = {h}" for h in h_values] + [""] * (nrows * ncols - len(h_values))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        shared_xaxes=True,
                        horizontal_spacing=0.07, vertical_spacing=0.12)
    for h_idx, h in enumerate(h_values):
        row, col = h_idx // ncols + 1, h_idx % ncols + 1
        is_first = (h_idx == 0)
        h_data = log_z_mcmc_for_graph.get(h, {})
        h_finals = log_z_final_for_graph.get(h, {})
        for b_idx, beta in enumerate(betas):
            color = PLOTLY_COLORS[b_idx % len(PLOTLY_COLORS)]
            beta_data = h_data.get(beta, {})
            if has_exact:
                exact_lz = h_finals.get(beta, {}).get("exact")
                if exact_lz is None:
                    continue
                denom = abs(exact_lz) if abs(exact_lz) > 1e-12 else 1.0
            for init in inits:
                for dyn in dyns:
                    if (init, dyn) not in beta_data:
                        continue
                    steps, lz = beta_data[(init, dyn)]
                    if has_exact:
                        y = np.maximum(np.abs(lz - exact_lz) / denom, 1e-10)
                    else:
                        y = lz
                    group = f"beta={beta} init={init} dyn={dyn}"
                    fig.add_trace(
                        go.Scatter(
                            x=steps, y=y, mode="lines",
                            name=f"β={beta}, {init}, {dyn}",
                            legendgroup=group,
                            showlegend=is_first,
                            line=dict(color=color, width=1.5,
                                      dash=STYLE_PLOTLY[(init, dyn)]),
                            hovertemplate=(
                                f"<b>β={beta}</b><br>"
                                f"init={init}<br>"
                                f"dynamics={dyn}<br>"
                                f"h={h}<br>"
                                "steps=%{x:,}<br>"
                                "y=%{y:.4g}<extra></extra>"
                            ),
                        ),
                        row=row, col=col,
                    )
        fig.update_xaxes(type="log", title_text="update steps per ⟨E⟩",
                         row=row, col=col)
        if has_exact:
            fig.update_yaxes(type="log",
                             title_text="|log Ẑ − log Z| / |log Z|",
                             row=row, col=col)
        else:
            fig.update_yaxes(title_text="log Ẑ (thermo)", row=row, col=col)
    fig.update_layout(
        title=dict(text=("log Z from MCMC <E>(β) via thermodynamic integration -- "
                         "relative error vs exact log Z (n ≤ 20)" if has_exact else
                         "log Z from MCMC <E>(β) via thermodynamic integration "
                         "(no exact reference for n>20)"),
                   x=0.5, xanchor="center", font=dict(size=12)),
        height=320 * nrows + 120,
        hovermode="closest",
        dragmode="pan",
        legend=dict(title="(β, init, dyn)", itemsizing="constant",
                    bgcolor="rgba(255,255,255,0.92)",
                    groupclick="togglegroup"),
        template="plotly_white",
    )
    return fig, has_exact


def _log_z_figure(graph_id: str, n: int,
                  log_z_budget_for_graph: Dict[float, Dict[float, Dict[int, List[Dict]]]],
                  log_z_final_for_graph: Dict[float, Dict[float, Dict[str, float]]],
                  betas: List[float], h_values: List[float]) -> Tuple[go.Figure, bool]:
    """FPRAS convergence: each curve is one (β, step_n_mult), plotting final
    log Ẑ across several independent FPRAS calls at varying `samples_per_segment`.
    x = total chain steps; y = relative error vs exact log Z when n ≤ 20
    (log-log), raw log Ẑ otherwise (log-x linear-y).
    Dash style encodes step_n_mult (solid = paper's 1/n; finer steps dashed/
    dotted); color encodes β."""
    has_exact = n <= 20
    ncols = 3
    nrows = math.ceil(len(h_values) / ncols)
    titles = [f"h = {h}" for h in h_values] + [""] * (nrows * ncols - len(h_values))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        shared_xaxes=True,
                        horizontal_spacing=0.07, vertical_spacing=0.14)
    for h_idx, h in enumerate(h_values):
        row, col = h_idx // ncols + 1, h_idx % ncols + 1
        is_first = (h_idx == 0)
        h_budget = log_z_budget_for_graph.get(h, {})
        h_finals = log_z_final_for_graph.get(h, {})
        for b_idx, beta in enumerate(betas):
            beta_data = h_budget.get(beta) or {}
            if not beta_data:
                continue
            color = PLOTLY_COLORS[b_idx % len(PLOTLY_COLORS)]
            if has_exact:
                exact_lz = h_finals.get(beta, {}).get("exact")
                if exact_lz is None:
                    continue
                ref_denom = abs(exact_lz) if abs(exact_lz) > 1e-12 else 1.0
            for step_n_mult in sorted(beta_data.keys()):
                recs = beta_data[step_n_mult]
                xs, ys, hovers = [], [], []
                for r in recs:
                    lz = r["log_Z"]
                    if math.isnan(lz):
                        continue
                    if has_exact:
                        y = max(abs(lz - exact_lz) / ref_denom, 1e-10)
                    else:
                        y = lz
                    xs.append(r["total_steps"])
                    ys.append(y)
                    hovers.append(
                        f"<b>β={beta}, h={h}</b><br>"
                        f"step = 1/({step_n_mult}n)<br>"
                        f"samples/segment = {r['m']:,}<br>"
                        f"n_segments = {r['n_segs']}<br>"
                        f"total steps = {r['total_steps']:,}<br>"
                        f"log Ẑ = {lz:.4f}"
                    )
                if not xs:
                    continue
                dash = STEP_N_MULT_DASH.get(step_n_mult, "solid")
                width = STEP_N_MULT_WIDTH.get(step_n_mult, 1.4)
                group = f"beta={beta} step={step_n_mult}"
                label = (f"β={beta}" if step_n_mult == 1
                         else f"β={beta}  (step 1/{step_n_mult}n)")
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines+markers",
                        name=label,
                        legendgroup=group,
                        showlegend=is_first,
                        line=dict(color=color, width=width, dash=dash),
                        marker=dict(color=color, size=7,
                                    line=dict(color="white", width=1)),
                        hovertext=hovers,
                        hoverinfo="text",
                    ),
                    row=row, col=col,
                )
        fig.update_xaxes(type="log", title_text="total FPRAS chain steps",
                         row=row, col=col)
        if has_exact:
            fig.update_yaxes(type="log",
                             title_text="|log Ẑ − log Z| / |log Z|",
                             row=row, col=col)
        else:
            fig.update_yaxes(title_text="log Ẑ", row=row, col=col)
    fig.update_layout(
        title=dict(text=("FPRAS convergence: relative error vs exact log Z, "
                         "sweeping samples_per_segment ∈ {100, 300, 1000, 3000, 10000}"
                         if has_exact else
                         "FPRAS log Ẑ vs total chain work, "
                         "sweeping samples_per_segment (no exact reference for n>20)"),
                   x=0.5, xanchor="center", font=dict(size=12)),
        height=320 * nrows + 120,
        hovermode="closest",
        dragmode="pan",
        legend=dict(title="β", itemsizing="constant",
                    bgcolor="rgba(255,255,255,0.92)",
                    groupclick="togglegroup"),
        template="plotly_white",
    )
    return fig, has_exact


# ---------- wall-time view (note 4): fair comparison of MCMC thermo and FPRAS ----------

# Microbenched on the same machine that produced these CSVs.  Both spin
# chains and the subgraphs chain are pure Python; numbers vary ~10% with n
# and β·h, so single representative values are used per kind.
#   Metropolis (simulate.py inline) : ~1.67 µs/step  =>  1.67 ms / 1000 steps
#   Glauber    (simulate.py inline) : ~1.90 µs/step  =>  1.90 ms / 1000 steps
#       (Glauber slightly slower because of the sigmoid; the rule does more
#        per step than Metropolis' single delta-energy check.)
#   JS subgraphs (SubgraphsChain)   : ~1.01 µs/step  =>  1.01 ms / 1000 steps
MCMC_US_PER_STEP = {"metropolis": 1.67, "glauber": 1.90}
JS_US_PER_STEP = 1.01


def _log_z_walltime_figure(graph_id: str, n: int,
                           log_z_mcmc_for_graph: Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]],
                           log_z_budget_for_graph: Dict[float, Dict[float, Dict[int, List[Dict]]]],
                           log_z_final_for_graph: Dict[float, Dict[float, Dict[str, float]]],
                           betas: List[float], h_values: List[float],
                           inits: List[str], dyns: List[str]
                           ) -> Tuple[go.Figure, bool]:
    """log Z from both estimators on a fair wall-time x-axis.

    For MCMC thermo at recorded step t for target beta_i with dynamics d:
      wall seconds = (i+1) * t * MCMC_US_PER_STEP[d] / 1e6
    -- (i+1) spin chains contribute to the integral up to beta_i, each at t
    steps, at the measured per-step rate for that dynamics.

    For FPRAS at (beta, m, step_n_mult):
      wall seconds = total_steps * JS_US_PER_STEP / 1e6
    -- total chain steps for that one full FPRAS run at the measured rate.

    Both shown on the same axes per h panel.  y is relative error vs exact
    log Z when n <= 20, raw log Z otherwise."""
    has_exact = n <= 20
    ncols = 3
    nrows = math.ceil(len(h_values) / ncols)
    titles = [f"h = {h}" for h in h_values] + [""] * (nrows * ncols - len(h_values))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        shared_xaxes=True, horizontal_spacing=0.07,
                        vertical_spacing=0.14)
    for h_idx, h in enumerate(h_values):
        row, col = h_idx // ncols + 1, h_idx % ncols + 1
        is_first = (h_idx == 0)
        h_mcmc = log_z_mcmc_for_graph.get(h, {})
        h_budget = log_z_budget_for_graph.get(h, {})
        h_finals = log_z_final_for_graph.get(h, {})
        for b_idx, beta in enumerate(betas):
            color = PLOTLY_COLORS[b_idx % len(PLOTLY_COLORS)]
            if has_exact:
                exact_lz = h_finals.get(beta, {}).get("exact")
                if exact_lz is None:
                    continue
                denom = abs(exact_lz) if abs(exact_lz) > 1e-12 else 1.0
            n_chains_for_target = b_idx + 1

            # MCMC thermo lines on the wall-time x-axis.
            beta_mcmc = h_mcmc.get(beta, {})
            for init in inits:
                for dyn in dyns:
                    if (init, dyn) not in beta_mcmc:
                        continue
                    steps, lz = beta_mcmc[(init, dyn)]
                    if has_exact:
                        y = np.maximum(np.abs(lz - exact_lz) / denom, 1e-10)
                    else:
                        y = lz
                    us_per_step = MCMC_US_PER_STEP.get(dyn, 2.0)
                    x_s = (steps * n_chains_for_target * us_per_step / 1e6)
                    group = f"beta={beta} init={init} dyn={dyn}"
                    fig.add_trace(
                        go.Scatter(
                            x=x_s, y=y, mode="lines",
                            name=f"MCMC β={beta}, {init}, {dyn}",
                            legendgroup=group, showlegend=is_first,
                            line=dict(color=color, width=1.4,
                                      dash=STYLE_PLOTLY[(init, dyn)]),
                            hovertemplate=(
                                f"<b>MCMC thermo β={beta}, h={h}</b><br>"
                                f"init={init}, dynamics={dyn}<br>"
                                f"n_chains used = {n_chains_for_target}<br>"
                                "wall time = %{x:.3g}s<br>"
                                "y = %{y:.4g}<extra></extra>"
                            ),
                        ),
                        row=row, col=col,
                    )

            # FPRAS markers on the wall-time x-axis.
            beta_budget = h_budget.get(beta, {})
            for step_n_mult in sorted(beta_budget.keys()):
                recs = beta_budget[step_n_mult]
                xs, ys, hovers = [], [], []
                for r in recs:
                    lz = r["log_Z"]
                    if math.isnan(lz):
                        continue
                    if has_exact:
                        yv = max(abs(lz - exact_lz) / denom, 1e-10)
                    else:
                        yv = lz
                    x_s = r["total_steps"] * JS_US_PER_STEP / 1e6
                    xs.append(x_s)
                    ys.append(yv)
                    hovers.append(
                        f"<b>FPRAS β={beta}, h={h}</b><br>"
                        f"step = 1/({step_n_mult}n)<br>"
                        f"samples/segment = {r['m']:,}<br>"
                        f"n_segments = {r['n_segs']}<br>"
                        f"chain steps = {r['total_steps']:,}<br>"
                        f"wall time = {x_s:.3g}s<br>"
                        f"log Ẑ = {lz:.4f}"
                    )
                if not xs:
                    continue
                symbol = STEP_N_MULT_MARKER.get(step_n_mult, "circle")
                dash = STEP_N_MULT_DASH.get(step_n_mult, "dot")
                group = f"beta={beta} step={step_n_mult}"
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines+markers",
                        name=("FPRAS β=" + str(beta)
                              + ("" if step_n_mult == 1
                                 else f"  step 1/{step_n_mult}n")),
                        legendgroup=group, showlegend=is_first,
                        line=dict(color=color, width=0.9, dash=dash),
                        marker=dict(color=color, size=9, symbol=symbol,
                                    line=dict(color="white", width=1.2)),
                        hovertext=hovers, hoverinfo="text",
                    ),
                    row=row, col=col,
                )
        fig.update_xaxes(type="log",
                         title_text="estimated wall time (s)",
                         row=row, col=col)
        if has_exact:
            fig.update_yaxes(type="log",
                             title_text="|log Ẑ − log Z| / |log Z|",
                             row=row, col=col)
        else:
            fig.update_yaxes(title_text="log Ẑ", row=row, col=col)
    fig.update_layout(
        title=dict(text=(
            "log Z vs estimated wall time -- MCMC thermo lines "
            f"({MCMC_US_PER_STEP['metropolis']:.2f} µs/step Metropolis, "
            f"{MCMC_US_PER_STEP['glauber']:.2f} µs/step Glauber); "
            f"JS FPRAS markers ({JS_US_PER_STEP:.2f} µs/step)"),
            x=0.5, xanchor="center", font=dict(size=11)),
        height=320 * nrows + 120,
        hovermode="closest",
        dragmode="pan",
        legend=dict(title="(β, source)", itemsizing="constant",
                    bgcolor="rgba(255,255,255,0.92)",
                    groupclick="togglegroup"),
        template="plotly_white",
    )
    return fig, has_exact


def _segments_table_html(graph_id: str,
                         log_z_budget_for_graph: Dict[float, Dict[float, Dict[int, List[Dict]]]],
                         betas: List[float], h_values: List[float]) -> str:
    """Render a small HTML table of n_segments at the paper's 1/n step
    (step_n_mult=1) for every (h, β) on this graph."""
    header = ('<tr><th>h \\ β</th>'
              + ''.join(f'<th>{b}</th>' for b in betas)
              + '</tr>')
    body_rows = []
    for h in h_values:
        cells = [f'<th>{h}</th>']
        for beta in betas:
            recs = (log_z_budget_for_graph.get(h, {})
                    .get(beta, {}).get(1) or [])
            cells.append(f'<td>{recs[0]["n_segs"]}</td>' if recs else '<td>-</td>')
        body_rows.append('<tr>' + ''.join(cells) + '</tr>')
    return (
        '<div class="segments-table-wrap" data-view="logz">'
        '<div class="segments-table-title">Schedule length (number of FPRAS '
        'segments) at each (h, β) for this graph -- paper\'s 1/n step:</div>'
        '<table class="segments-table"><thead>' + header + '</thead>'
        '<tbody>' + ''.join(body_rows) + '</tbody></table></div>'
    )


# ---------- HTML page assembly ----------

PAGE_STYLE = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif;
         margin: 18px; color: #222; line-height: 1.45; }
  h1 { margin: 0 0 4px 0; font-size: 22px; }
  h2 { margin: 24px 0 6px; font-size: 17px; border-top: 1px solid #ccc;
       padding-top: 14px; }
  .intro { color: #333; font-size: 13.5px; margin: 0 0 14px 0;
           max-width: 880px; }
  .intro code { background: #f1f1f1; padding: 1px 5px; border-radius: 3px;
                font-size: 90%; }
  .intro details { margin-top: 6px; }
  .intro summary { cursor: pointer; color: #1166bb; font-weight: 600; }
  .controls { display: flex; flex-wrap: wrap; gap: 16px;
              border: 1px solid #ddd; border-radius: 8px;
              padding: 10px 12px; margin: 12px 0;
              background: #fafafa; align-items: flex-start; }
  .controls fieldset { border: none; padding: 0; margin: 0; }
  .controls legend { font-weight: 600; font-size: 12px;
                     text-transform: uppercase; color: #444;
                     letter-spacing: 0.5px; margin-bottom: 4px; }
  .controls label { display: inline-block; margin: 2px 8px 2px 0;
                    font-size: 13px; cursor: pointer; user-select: none; }
  .controls .shortcut-btns { display: flex; gap: 6px; margin-top: 4px; }
  .controls button { font-size: 12px; padding: 2px 8px; cursor: pointer;
                     border: 1px solid #bbb; border-radius: 4px;
                     background: white; }
  .controls button:hover { background: #eee; }
  .graph-section { margin-bottom: 10px; }
  .graph-section.hidden { display: none; }
  .graphfig { margin: 8px 0; }
  .view-hidden { display: none; }
  .segments-table-wrap { margin: 10px 0 6px 0; }
  .segments-table-title { font-size: 12.5px; color: #444; margin-bottom: 4px; }
  .segments-table { border-collapse: collapse; font-size: 12px; font-family: ui-monospace, Menlo, monospace; }
  .segments-table th, .segments-table td { border: 1px solid #ccc; padding: 2px 8px; text-align: center; }
  .segments-table th { background: #f1f1f1; font-weight: 600; }
  .segments-table tbody th { background: #f7f7f7; }
</style>
"""

INTRO_HTML = """
<h1>Single-site Ising MCMC convergence on 3-regular graphs</h1>
<p class="intro">
Each chain is started from one of two initial distributions over spins, and run
under one of two single-site update rules.  All combinations of
(β, h, init, dynamics) are sweep-tested per graph; the curves show the
running-average energy <i>Ê<sub>t</sub></i> as a function of update count.
For small graphs (n=16) we brute-force the exact ⟨E⟩<sub>β,h</sub> and plot
relative error; for n=40 we plot Ê<sub>t</sub> directly.
<details>
<summary>What's the difference between Metropolis and Glauber?</summary>
<p>Both are single-site reversible Markov chains with the Gibbs measure
e<sup>−βH</sup>/Z as their stationary distribution; they differ only in the
update rule applied to a uniformly chosen site v.</p>
<ul>
<li><b>Metropolis</b>: <i>propose</i> flipping σ<sub>v</sub>; accept with
probability min(1, e<sup>−β·ΔE</sup>).  Every favorable flip (ΔE ≤ 0) is
accepted with probability 1, so Metropolis is "more aggressive" downhill.</li>
<li><b>Glauber (heat-bath)</b>: <i>resample</i> σ<sub>v</sub> from its
conditional distribution given its neighbors.  Concretely, σ<sub>v</sub> ← +1
with probability σ(2β·h<sub>v,eff</sub>) where
h<sub>v,eff</sub> = (sum of neighbor spins) + h and σ is the logistic
sigmoid.  There is no propose-then-reject step.  Even a favorable flip
happens with probability strictly less than 1, so Glauber is "smoother".</li>
</ul>
<p>The two dynamics have the same stationary distribution but different
mixing properties; mixing-time results in the literature are typically
stated for Glauber.</p>
<p>The two initial distributions are:
<i>ground</i> (uniform over the two all-aligned ground states +1<sup>n</sup>
and −1<sup>n</sup>) and <i>uniform</i> (each spin independently +1 or −1).
</p>
</details>
<details>
<summary>What are the log Z views?</summary>
<p>Two separate views, each with 5 h panels.</p>
<ul>
<li><b>log Z (MCMC thermo)</b> — same spin chains as the energy view,
post-processed by trapezoidal integration of ⟨E⟩ over β
(<code>log Z(β,h) = n·log 2 − ∫₀^β ⟨E⟩(β',h) dβ'</code>).  One line per
(β, init, dyn); x = per-chain steps.</li>
<li><b>log Z (JS FPRAS budget)</b> — each marker is one independent full
FPRAS run at fixed <code>samples_per_segment ∈ {100,300,1000,3000,10000}</code>;
lines connect runs at the same β.  Dash style encodes the schedule step
(<code>1/n</code> solid, <code>1/(4n)</code> dashed, <code>1/(20n)</code> dotted).
x = total FPRAS chain steps = <code>n_segments × (burnin + samples_per_segment)</code>.
Schedule-length table below the chart.</li>
</ul>
<p>For n=16 the y axis is relative error vs exact log Z (log-log); for n=40 raw log Ẑ.</p>
</details>
<details>
<summary>What is the "log Z vs wall time" view?</summary>
<p>Both estimators replotted with x = estimated wall time (seconds), so you
can fairly compare which one reaches a target accuracy faster.  Step rates
were microbenched on this machine:
Metropolis ≈ 1.67 µs/step, Glauber ≈ 1.90 µs/step, JS subgraphs ≈ 1.01 µs/step.
Each MCMC curve has x = (chains_used_for_target_β) · per_chain_steps · µs/step,
each FPRAS marker has x = total_FPRAS_chain_steps · 1.01 µs.</p>
</details>
</p>
"""

CONTROLS_HTML_TEMPLATE = """
<div class="controls">
  <fieldset id="ctl-view">
    <legend>View</legend>
    <label><input type="radio" name="viewsel" checked value="energy"> energy convergence</label>
    <label><input type="radio" name="viewsel" value="logz_mcmc"> log Z (MCMC thermo)</label>
    <label><input type="radio" name="viewsel" value="logz_fpras"> log Z (JS FPRAS budget)</label>
    <label><input type="radio" name="viewsel" value="logz_walltime"> log Z vs wall time</label>
  </fieldset>
  <fieldset id="ctl-graph">
    <legend>Graph</legend>
    {graph_radios}
  </fieldset>
  <fieldset id="ctl-init" class="view-only-energy">
    <legend>Initial distribution</legend>
    <label><input type="checkbox" data-filter="init" data-value="ground"> ground</label>
    <label><input type="checkbox" checked data-filter="init" data-value="uniform"> uniform</label>
    <div class="shortcut-btns">
      <button type="button" onclick="setAll('init', true)">all</button>
      <button type="button" onclick="setAll('init', false)">none</button>
    </div>
  </fieldset>
  <fieldset id="ctl-dyn" class="view-only-energy">
    <legend>Dynamics</legend>
    <label><input type="checkbox" data-filter="dyn" data-value="metropolis"> metropolis</label>
    <label><input type="checkbox" checked data-filter="dyn" data-value="glauber"> glauber</label>
    <div class="shortcut-btns">
      <button type="button" onclick="setAll('dyn', true)">all</button>
      <button type="button" onclick="setAll('dyn', false)">none</button>
    </div>
  </fieldset>
  <fieldset id="ctl-beta">
    <legend>Beta</legend>
    {beta_checkboxes}
    <div class="shortcut-btns">
      <button type="button" onclick="setAll('beta', true)">all</button>
      <button type="button" onclick="setAll('beta', false)">none</button>
    </div>
  </fieldset>
</div>
"""

CONTROLS_SCRIPT = """
<script>
  // legendgroup encoding: energy traces use "beta=<x> init=<y> dyn=<z>",
  // log Z traces use "method=<m>".
  function parseGroup(g) {
    if (!g) return null;
    const out = {};
    g.split(' ').forEach(kv => {
      const i = kv.indexOf('=');
      if (i < 0) return;
      out[kv.slice(0, i)] = kv.slice(i + 1);
    });
    return out;
  }
  function currentView() {
    const v = document.querySelector('input[name="viewsel"]:checked');
    return v ? v.value : 'energy';
  }
  // Init/dyn filters apply in any view that contains MCMC traces.
  const VIEWS_WITH_INIT_DYN = new Set(['energy', 'logz_mcmc', 'logz_walltime']);
  const PLOT_PREFIXES = ['convfig-', 'logzmcmcfig-', 'logzfig-', 'logzwalltimefig-'];

  function applyTraceFilters() {
    const initSet = new Set(Array.from(document.querySelectorAll('[data-filter="init"]:checked')).map(e => e.dataset.value));
    const dynSet = new Set(Array.from(document.querySelectorAll('[data-filter="dyn"]:checked')).map(e => e.dataset.value));
    const betaSet = new Set(Array.from(document.querySelectorAll('[data-filter="beta"]:checked')).map(e => e.dataset.value));
    const sel = document.querySelector('input[name="graphsel"]:checked');
    if (!sel) return;
    PLOT_PREFIXES.forEach(prefix => {
      const div = document.getElementById(prefix + sel.value);
      if (!div || !div.data) return;
      const visibility = div.data.map(trace => {
        const m = parseGroup(trace.legendgroup);
        if (!m) return true;
        if (m.init !== undefined && !initSet.has(m.init)) return false;
        if (m.dyn !== undefined && !dynSet.has(m.dyn)) return false;
        if (m.beta !== undefined && !betaSet.has(m.beta)) return false;
        return true;
      });
      Plotly.restyle(div, {visible: visibility});
    });
  }
  function applyView() {
    const view = currentView();
    document.querySelectorAll('[data-view]').forEach(div => {
      if (div.dataset.view === view) div.classList.remove('view-hidden');
      else div.classList.add('view-hidden');
    });
    document.querySelectorAll('.view-only-energy').forEach(el => {
      if (VIEWS_WITH_INIT_DYN.has(view)) el.classList.remove('view-hidden');
      else el.classList.add('view-hidden');
    });
    setTimeout(() => {
      PLOT_PREFIXES.forEach(prefix => {
        const div = document.querySelector('.graph-section:not(.hidden) [data-view]:not(.view-hidden) [id^="' + prefix + '"]');
        if (div && window.Plotly && div.layout) Plotly.Plots.resize(div);
      });
    }, 50);
  }
  function applyGraphSelection() {
    const sel = document.querySelector('input[name="graphsel"]:checked');
    const which = sel ? sel.value : null;
    document.querySelectorAll('.graph-section').forEach(sec => {
      if (sec.dataset.graph === which) {
        sec.classList.remove('hidden');
      } else {
        sec.classList.add('hidden');
      }
    });
    setTimeout(() => {
      ['graphfig-'].concat(PLOT_PREFIXES).forEach(prefix => {
        document.querySelectorAll('.graph-section:not(.hidden) [id^="' + prefix + '"]').forEach(div => {
          if (window.Plotly && div.layout) Plotly.Plots.resize(div);
        });
      });
      applyTraceFilters();
      applyView();
    }, 50);
  }
  document.querySelectorAll('.controls input[type=checkbox]').forEach(cb => {
    cb.addEventListener('change', applyTraceFilters);
  });
  document.querySelectorAll('input[name="graphsel"]').forEach(r => {
    r.addEventListener('change', applyGraphSelection);
  });
  document.querySelectorAll('input[name="viewsel"]').forEach(r => {
    r.addEventListener('change', applyView);
  });
  function setAll(filter, value) {
    document.querySelectorAll('[data-filter="' + filter + '"]').forEach(cb => cb.checked = value);
    applyTraceFilters();
  }
  document.addEventListener('DOMContentLoaded', () => {
    applyGraphSelection();
    applyView();
  });
</script>
"""


def _make_section(graph_id: str, n: int, G_nx: nx.Graph, graph_data: Dict,
                  exact: Dict[Tuple[str, float, float], float],
                  log_z_for_graph: Dict[float, Dict[float, Dict[str, float]]],
                  log_z_budget_for_graph: Dict[float, Dict[float, Dict[int, List[Dict]]]],
                  log_z_mcmc_for_graph: Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]],
                  betas: List[float], h_values: List[float],
                  inits: List[str], dyns: List[str],
                  is_default_visible: bool) -> str:
    graph_fig = _graph_figure(G_nx, title=f"{graph_id} — spring layout")
    conv_fig, _ = _convergence_figure(graph_id, graph_data, exact,
                                      betas, h_values, inits, dyns)
    log_z_mcmc_fig, _ = _log_z_mcmc_figure(
        graph_id, log_z_mcmc_for_graph, log_z_for_graph,
        betas, h_values, inits, dyns,
    )
    log_z_fig, _ = _log_z_figure(
        graph_id, n, log_z_budget_for_graph, log_z_for_graph,
        betas, h_values,
    )
    log_z_walltime_fig, _ = _log_z_walltime_figure(
        graph_id, n, log_z_mcmc_for_graph, log_z_budget_for_graph,
        log_z_for_graph, betas, h_values, inits, dyns,
    )
    seg_table_html = _segments_table_html(graph_id, log_z_budget_for_graph,
                                          betas, h_values)
    graph_html = graph_fig.to_html(include_plotlyjs=False, full_html=False,
                                   div_id=f"graphfig-{graph_id}")
    conv_html = conv_fig.to_html(include_plotlyjs=False, full_html=False,
                                 div_id=f"convfig-{graph_id}")
    log_z_mcmc_html = log_z_mcmc_fig.to_html(include_plotlyjs=False, full_html=False,
                                             div_id=f"logzmcmcfig-{graph_id}")
    log_z_html = log_z_fig.to_html(include_plotlyjs=False, full_html=False,
                                   div_id=f"logzfig-{graph_id}")
    log_z_walltime_html = log_z_walltime_fig.to_html(include_plotlyjs=False, full_html=False,
                                                     div_id=f"logzwalltimefig-{graph_id}")
    graph_wrap = f'<div class="graphfig">{graph_html}</div>'
    conv_wrap = f'<div class="convfig" data-view="energy">{conv_html}</div>'
    log_z_mcmc_wrap = (f'<div class="logzmcmcfig" data-view="logz_mcmc">'
                       f'{log_z_mcmc_html}</div>')
    log_z_wrap = (f'<div class="logzfig" data-view="logz_fpras">'
                  f'{log_z_html}{seg_table_html}</div>')
    log_z_walltime_wrap = (f'<div class="logzwalltimefig" data-view="logz_walltime">'
                           f'{log_z_walltime_html}</div>')
    section_class = "graph-section" + ("" if is_default_visible else " hidden")
    return (
        f'<section class="{section_class}" data-graph="{graph_id}">'
        f'<h2>{graph_id}  (3-regular, n={n})</h2>'
        + graph_wrap + conv_wrap + log_z_mcmc_wrap
        + log_z_wrap + log_z_walltime_wrap +
        '</section>'
    )


def render_combined_html(out_path: str,
                         data: Dict, graph_ids: List[str],
                         exact: Dict[Tuple[str, float, float], float],
                         log_z: Dict[str, Dict[float, Dict[float, Dict[str, float]]]],
                         log_z_budget: Dict[str, Dict[float, Dict[float, Dict[int, List[Dict]]]]],
                         log_z_mcmc: Dict[str, Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]]],
                         betas: List[float], h_values: List[float],
                         inits: List[str], dyns: List[str]) -> None:
    # Default-selected graph is the first one (n16_graph0 by ordering).
    default_selected = graph_ids[0] if graph_ids else None

    graph_radios = "".join(
        f'<label><input type="radio" name="graphsel"'
        f'{" checked" if gid == default_selected else ""} value="{gid}"> {gid}</label>'
        for gid in graph_ids
    )
    default_beta = 0.5
    beta_checkboxes = "".join(
        f'<label><input type="checkbox"{" checked" if b == default_beta else ""} '
        f'data-filter="beta" data-value="{b}"> {b}</label>'
        for b in betas
    )
    controls_html = CONTROLS_HTML_TEMPLATE.format(
        graph_radios=graph_radios,
        beta_checkboxes=beta_checkboxes,
    )

    # We need plotly.js once on the page; let the first chart load it.
    sections: List[str] = []
    for gid in graph_ids:
        G_nx = load_graph(gid)
        n = G_nx.number_of_nodes()
        sections.append(_make_section(
            gid, n, G_nx, data[gid], exact,
            log_z.get(gid, {}), log_z_budget.get(gid, {}),
            log_z_mcmc.get(gid, {}),
            betas, h_values, inits, dyns,
            is_default_visible=(gid == default_selected),
        ))
    # Inject plotly.js as the very first script via a stub chart helper or just
    # add a CDN <script>.  Using the CDN tag directly is the simplest.
    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Ising MCMC convergence</title>"
        + PAGE_STYLE
        + plotly_cdn
        + "</head><body>"
        + INTRO_HTML
        + controls_html
        + "".join(sections)
        + CONTROLS_SCRIPT
        + "</body></html>"
    )
    with open(out_path, "w") as f:
        f.write(html)


# ---------- driver ----------

def main():
    data, graph_ids, betas, h_values, inits, dyns = load_traces(TRACES_CSV)
    exact = load_exact(EXACT_CSV)
    log_z = load_log_z(LOG_Z_CSV)
    log_z_budget = load_log_z_budget(LOG_Z_BUDGET_CSV)
    log_z_mcmc = load_log_z_mcmc(LOG_Z_MCMC_CSV)
    print(f"loaded {len(graph_ids)} graphs: {graph_ids}")
    print(f"  betas={betas}\n  h={h_values}\n  inits={inits}\n  dyns={dyns}")
    print(f"  log_z graphs:        {sorted(log_z.keys())}")
    print(f"  log_z budget graphs: {sorted(log_z_budget.keys())}")
    print(f"  log_z mcmc graphs:   {sorted(log_z_mcmc.keys())}")

    # Per-graph PNG (archival).
    for graph_id in graph_ids:
        G_nx = load_graph(graph_id)
        n = G_nx.number_of_nodes()
        png = f"convergence_{graph_id}.png"
        static_plot(graph_id, n, data[graph_id], exact, betas, h_values,
                    inits, dyns, png)
        print(f"  wrote {png}")

    # One combined interactive page.
    render_combined_html(COMBINED_HTML, data, graph_ids, exact, log_z,
                         log_z_budget, log_z_mcmc,
                         betas, h_values, inits, dyns)
    print(f"  wrote {COMBINED_HTML}")


if __name__ == "__main__":
    main()
