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

import run_fep  # for FEP_BURNIN (kept in sync with the producer)
FEP_BURNIN = run_fep.BURNIN


DATA_DIR = "data"
GRAPHS_SUBDIR = os.path.join(DATA_DIR, "graphs")
TRACES_CSV = os.path.join(DATA_DIR, "traces.csv")
EXACT_CSV = os.path.join(DATA_DIR, "exact.csv")
LOG_Z_CSV = os.path.join(DATA_DIR, "log_z.csv")
LOG_Z_JS_SWEEP_CSV = os.path.join(DATA_DIR, "log_z_js_sweep.csv")
LOG_Z_MCMC_CSV = os.path.join(DATA_DIR, "log_z_mcmc.csv")
LOG_Z_FEP_CSV = os.path.join(DATA_DIR, "log_z_fep.csv")
LOG_Z_TAYLOR_CSV = os.path.join(DATA_DIR, "log_z_taylor.csv")
# n above which brute-force exact is intractable; for these graphs the log-Z
# relative-error reference is the FEP estimate (see main()).
EXACT_MAX_N = 20
VENDOR_DIR = "vendor"
PLOTLY_LOCAL_JS = os.path.join(VENDOR_DIR, "plotly.min.js")
PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _fig_lazy_html(fig: go.Figure, div_id: str) -> str:
    """Emit an empty div + a `<script type="application/json">` carrying the
    figure spec.  The JS in CONTROLS_SCRIPT renders each figure with
    Plotly.newPlot only when its section/view becomes visible, which keeps
    initial page load from running Plotly on 16 * 5 charts at once."""
    spec = fig.to_json()
    spec = spec.replace("</", "<\\/")
    return (
        f'<div id="{div_id}" class="plotly-lazy"></div>'
        f'<script type="application/json" data-figdiv="{div_id}">'
        f'{spec}</script>'
    )


def _ensure_plotly_js() -> None:
    """Fetch plotly.min.js into VENDOR_DIR once, so convergence.html can be
    viewed offline.  No-op if the file already exists."""
    if os.path.exists(PLOTLY_LOCAL_JS):
        return
    os.makedirs(VENDOR_DIR, exist_ok=True)
    import urllib.request
    print(f"  fetching {PLOTLY_CDN_URL} -> {PLOTLY_LOCAL_JS}")
    urllib.request.urlretrieve(PLOTLY_CDN_URL, PLOTLY_LOCAL_JS)
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
# Display labels for the two init distributions, by temperature:
# "ground" = all-aligned (zero-T) start -> low-temp;
# "uniform" = random spins (infinite-T / beta=0) start -> high-temp.
# These rename only what the user sees; "ground"/"uniform" stay as the
# internal data keys (CSV values, STYLE_* dict keys, filter values).
INIT_LABEL = {"ground": "low-temp", "uniform": "high-temp"}
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


def load_log_z_fep(path: str
                   ) -> Dict[str, Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]]]:
    """log_z_fep[graph_id][h][beta][(init, dyn)] = (steps_array, log_z_array).
    Free-energy-perturbation telescoping estimate from run_fep.py, one series
    per (graph, h, beta, init, dynamics)."""
    out: Dict = {}
    if not os.path.exists(path):
        return out
    raw = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["graph_id"], float(row["h"]), float(row["beta"]),
                   row["init"], row["dynamics"])
            raw[key].append((int(row["step"]), float(row["log_Z"])))
    for (gid, h, beta, init, dyn), pairs in raw.items():
        pairs.sort()
        steps = np.array([p[0] for p in pairs])
        lz = np.array([p[1] for p in pairs])
        (out.setdefault(gid, {}).setdefault(h, {})
            .setdefault(beta, {})[(init, dyn)]) = (steps, lz)
    return out


def load_log_z_js_sweep(path: str
                      ) -> Dict[str, Dict[float, Dict[float, Dict[int, List[Dict]]]]]:
    """log_z_js_sweep[graph_id][h][beta][step_n_mult] = [
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


def load_log_z_taylor(path: str
                   ) -> Dict[str, Dict[float, Dict[float, Dict[str, List[Dict]]]]]:
    """log_z_taylor[graph_id][h][beta][method] = [{m, log_Z, log_Z_exact, rel_err,
        bound, runtime_s}, ...]  sorted by m ascending.  Missing/timed-out
    cells appear as NaN in log_Z / rel_err."""
    out: Dict = {}
    if not os.path.exists(path):
        return out

    def _maybe_float(s: str) -> float:
        try:
            return float(s)
        except (ValueError, TypeError):
            return float("nan")

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gid = row["graph_id"]
            h = float(row["h"])
            beta = float(row["beta"])
            method = row["method"]
            rec = dict(
                m=int(row["m"]),
                log_Z=_maybe_float(row["log_Z"]),
                log_Z_exact=_maybe_float(row["log_Z_exact"]),
                rel_err=_maybe_float(row["rel_err"]),
                bound=_maybe_float(row["error_bound"]),
                runtime_s=_maybe_float(row["runtime_s"]),
            )
            (out.setdefault(gid, {}).setdefault(h, {})
                .setdefault(beta, {}).setdefault(method, []).append(rec))
    for gid in out:
        for h in out[gid]:
            for beta in out[gid][h]:
                for method in out[gid][h][beta]:
                    out[gid][h][beta][method].sort(key=lambda r: r["m"])
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
                   label="low-temp / metropolis"),
        plt.Line2D([], [], color="black", linewidth=1.8, linestyle=(0, (6, 2)),
                   label="high-temp / metropolis"),
        plt.Line2D([], [], color="black", linewidth=1.8, linestyle=":",
                   label="low-temp / glauber"),
        plt.Line2D([], [], color="black", linewidth=1.8, linestyle="-.",
                   label="high-temp / glauber"),
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
                            name=f"β={beta}, {INIT_LABEL[init]}, {dyn}",
                            legendgroup=group,
                            showlegend=is_first,
                            line=dict(color=color, width=1.6,
                                      dash=STYLE_PLOTLY[(init, dyn)]),
                            hovertemplate=(
                                f"<b>β={beta}</b><br>"
                                f"init={INIT_LABEL[init]}<br>"
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
                           log_z_js_sweep_for_graph: Dict[float, Dict[float, Dict[int, List[Dict]]]],
                           log_z_final_for_graph: Dict[float, Dict[float, Dict[str, float]]],
                           betas: List[float], h_values: List[float],
                           inits: List[str], dyns: List[str]
                           ) -> Tuple[go.Figure, bool]:
    """Two sub-panels per h, stacked: MCMC + thermo integration on top,
    Jerrum-Sinclair sweep below.  Both use x = total chain steps so the
    panels are directly comparable.
      - MCMC thermo at recorded step t for target beta_i:
            x = (i+1) * t   (i+1 spin chains contribute to the integral)
      - FPRAS at (beta, m, step_n_mult): x = total_steps from the CSV.
    y is relative error vs the reference log Z (brute-force for n <= 20,
    long-Glauber thermo for larger n) when available, raw log Z otherwise."""
    has_exact = any("exact" in log_z_final_for_graph.get(h, {}).get(beta, {})
                    for h in h_values for beta in betas)
    nrows = 2 * len(h_values)
    titles: List[str] = []
    for h in h_values:
        titles.append(f"h = {h}  ·  MCMC + thermodynamic integration")
        titles.append(f"h = {h}  ·  Jerrum-Sinclair sweep")
    fig = make_subplots(rows=nrows, cols=1, subplot_titles=titles,
                        shared_xaxes=False,
                        vertical_spacing=0.04)
    for h_idx, h in enumerate(h_values):
        mcmc_row = 2 * h_idx + 1
        fpras_row = 2 * h_idx + 2
        is_first = (h_idx == 0)
        h_mcmc = log_z_mcmc_for_graph.get(h, {})
        h_js_sweep = log_z_js_sweep_for_graph.get(h, {})
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
                            name=f"β={beta}, {INIT_LABEL[init]}, {dyn}",
                            legendgroup=group, showlegend=is_first,
                            line=dict(color=color, width=1.4,
                                      dash=STYLE_PLOTLY[(init, dyn)]),
                            hovertemplate=(
                                f"<b>β={beta}, h={h}</b><br>"
                                f"init={INIT_LABEL[init]}, dynamics={dyn}<br>"
                                "total chain steps=%{x:,}<br>"
                                "y=%{y:.4g}<extra></extra>"
                            ),
                        ),
                        row=mcmc_row, col=1,
                    )

            # FPRAS budget on bottom sub-panel.
            beta_js_sweep = h_js_sweep.get(beta, {})
            for step_n_mult in sorted(beta_js_sweep.keys()):
                recs = beta_js_sweep[step_n_mult]
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
        title=dict(text="log Z: MCMC+thermo (top) vs Jerrum-Sinclair sweep (bottom), per h",
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


def _log_z_mcmc_figure(graph_id: str, n: int,
                       log_z_mcmc_for_graph: Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]],
                       log_z_final_for_graph: Dict[float, Dict[float, Dict[str, float]]],
                       betas: List[float], h_values: List[float],
                       inits: List[str], dyns: List[str],
                       show_rel: bool) -> Tuple[go.Figure, bool]:
    """log Z derived from the spin chains' <E>(beta, step) traces via
    trapezoidal thermodynamic integration over beta.  Same 4 (init, dyn) x
    10 beta structure as the energy view.  show_rel=False -> raw log Ẑ;
    show_rel=True -> relative error vs the reference (exact log Z for
    n<=EXACT_MAX_N, the FEP estimate for larger n)."""
    has_exact = show_rel
    ref_name = "exact log Z" if n <= EXACT_MAX_N else "FEP reference"
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
                            name=f"β={beta}, {INIT_LABEL[init]}, {dyn}",
                            legendgroup=group, showlegend=is_first,
                            line=dict(color=color, width=1.5,
                                      dash=STYLE_PLOTLY[(init, dyn)]),
                            hovertemplate=(
                                f"<b>β={beta}</b><br>init={INIT_LABEL[init]}<br>"
                                f"dynamics={dyn}<br>h={h}<br>steps=%{{x:,}}<br>"
                                "y=%{y:.4g}<extra></extra>"),
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
        title=dict(text=("log Z from MCMC ⟨E⟩(β) via thermodynamic integration — "
                         + (f"relative error vs {ref_name}" if has_exact
                            else "raw log Ẑ over steps")),
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


def _log_z_fep_figure(graph_id: str, n: int,
                      fep_for_graph: Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]],
                      log_z_final_for_graph: Dict[float, Dict[float, Dict[str, float]]],
                      betas: List[float], h_values: List[float],
                      inits: List[str], dyns: List[str],
                      show_rel: bool) -> go.Figure:
    """Free-energy-perturbation (Zwanzig) telescoping estimate of log Z vs
    chain work:  log Z(b_{k+1}) = n*log2 + sum_j log < e^{-Db_j E} >_{b_j}.
    Unlike trapezoidal integration of <E> it carries no grid-discretization
    bias.  One line per (β, init, dyn) -- same toggles as the energy view.
    show_rel=False -> raw log Ẑ; show_rel=True -> relative error vs the
    reference (exact log Z for n<=EXACT_MAX_N, the FEP estimate itself for
    larger n)."""
    ref_name = "exact log Z" if n <= EXACT_MAX_N else "FEP reference"
    ncols = 3
    nrows = math.ceil(len(h_values) / ncols)
    titles = [f"h = {h}" for h in h_values] + [""] * (nrows * ncols - len(h_values))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        shared_xaxes=True,
                        horizontal_spacing=0.07, vertical_spacing=0.12)
    for h_idx, h in enumerate(h_values):
        row, col = h_idx // ncols + 1, h_idx % ncols + 1
        is_first = (h_idx == 0)
        h_data = fep_for_graph.get(h, {})
        h_finals = log_z_final_for_graph.get(h, {})
        for b_idx, beta in enumerate(betas):
            beta_data = h_data.get(beta, {})
            if not beta_data:
                continue
            color = PLOTLY_COLORS[b_idx % len(PLOTLY_COLORS)]
            # Total chain work to reach this β: it telescopes through the
            # (b_idx+1) ladder segments below it, each running FEP_BURNIN +
            # (recorded sample count).  Reported like the JS total_steps so
            # the two are directly comparable.
            n_seg = b_idx + 1
            ref_lz = h_finals.get(beta, {}).get("exact")
            if show_rel:
                if ref_lz is None:
                    continue
                denom = abs(ref_lz) if abs(ref_lz) > 1e-12 else 1.0
            for init in inits:
                for dyn in dyns:
                    if (init, dyn) not in beta_data:
                        continue
                    steps, lz = beta_data[(init, dyn)]
                    x = n_seg * (FEP_BURNIN + steps)
                    if show_rel:
                        y = np.maximum(np.abs(lz - ref_lz) / denom, 1e-10)
                    else:
                        y = lz
                    fig.add_trace(
                        go.Scatter(
                            x=x, y=y, mode="lines",
                            name=f"β={beta}, {INIT_LABEL[init]}, {dyn}",
                            legendgroup=f"beta={beta} init={init} dyn={dyn}",
                            showlegend=is_first,
                            line=dict(color=color, width=1.5,
                                      dash=STYLE_PLOTLY[(init, dyn)]),
                            hovertemplate=(
                                f"<b>β={beta}, h={h}</b><br>init={INIT_LABEL[init]}, "
                                f"dyn={dyn}<br>total chain steps=%{{x:,}}<br>"
                                + ("rel err=%{y:.4g}" if show_rel else "log Ẑ=%{y:.4f}")
                                + "<extra></extra>"),
                        ),
                        row=row, col=col,
                    )
        fig.update_xaxes(type="log",
                         title_text="total chain steps (burn-in + samples, all segments)",
                         row=row, col=col)
        if show_rel:
            fig.update_yaxes(type="log",
                             title_text="|log Ẑ − log Z| / |log Z|",
                             row=row, col=col)
        else:
            fig.update_yaxes(title_text="log Ẑ", row=row, col=col)
    fig.update_layout(
        title=dict(text=("log Z via FEP telescoping — "
                         + (f"relative error vs {ref_name}" if show_rel
                            else "raw log Ẑ over chain work")),
                   x=0.5, xanchor="center", font=dict(size=12)),
        height=320 * nrows + 120,
        hovermode="closest", dragmode="pan",
        legend=dict(title="(β, init, dyn)", itemsizing="constant",
                    bgcolor="rgba(255,255,255,0.92)",
                    groupclick="togglegroup"),
        template="plotly_white",
    )
    return fig


def _log_z_figure(graph_id: str, n: int,
                  log_z_js_sweep_for_graph: Dict[float, Dict[float, Dict[int, List[Dict]]]],
                  log_z_final_for_graph: Dict[float, Dict[float, Dict[str, float]]],
                  betas: List[float], h_values: List[float],
                  show_rel: bool) -> Tuple[go.Figure, bool]:
    """FPRAS convergence: each curve is one (β, step_n_mult), plotting final
    log Ẑ across several independent FPRAS calls at varying
    `samples_per_segment`.  x = total chain steps; show_rel=False -> raw
    log Ẑ, show_rel=True -> relative error vs the reference (exact log Z for
    n<=EXACT_MAX_N, the FEP estimate for larger n).
    Dash style encodes step_n_mult (solid = paper's 1/n; finer steps dashed/
    dotted); color encodes β."""
    has_exact = show_rel
    ref_name = "exact log Z" if n <= EXACT_MAX_N else "FEP reference"
    ncols = 3
    nrows = math.ceil(len(h_values) / ncols)
    titles = [f"h = {h}" for h in h_values] + [""] * (nrows * ncols - len(h_values))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        shared_xaxes=True,
                        horizontal_spacing=0.07, vertical_spacing=0.14)
    for h_idx, h in enumerate(h_values):
        row, col = h_idx // ncols + 1, h_idx % ncols + 1
        is_first = (h_idx == 0)
        h_js_sweep = log_z_js_sweep_for_graph.get(h, {})
        h_finals = log_z_final_for_graph.get(h, {})
        for b_idx, beta in enumerate(betas):
            beta_data = h_js_sweep.get(beta) or {}
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
                        x=xs, y=ys, mode="lines+markers", name=label,
                        legendgroup=group, showlegend=is_first,
                        line=dict(color=color, width=width, dash=dash),
                        marker=dict(color=color, size=7,
                                    line=dict(color="white", width=1)),
                        hovertext=hovers, hoverinfo="text",
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
        title=dict(text=("FPRAS, sweeping samples_per_segment ∈ "
                         "{100, 300, 1000, 3000, 10000} — "
                         + (f"relative error vs {ref_name}" if has_exact
                            else "raw log Ẑ vs total chain work")),
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

# Microbenched per graph size on the machine that produced these CSVs
# (run_microbench.py, measured with nothing else running).  Per-step cost
# does drift a little with n, so we keep a number per size and pick the
# nearest measured size for the graph being plotted.
#   { n : {"metropolis": µs, "glauber": µs, "js": µs} }
US_PER_STEP_BY_N = {
    16: {"metropolis": 1.547, "glauber": 1.421, "js": 0.553},
    30: {"metropolis": 1.407, "glauber": 1.344, "js": 0.547},
    40: {"metropolis": 1.46, "glauber": 1.381, "js": 0.547},
}

# FEP per-step cost (single-site update + online log-sum-exp), microbenched
# per size and dynamics by run_microbench_fep.py.
FEP_US_PER_STEP_BY_N = {
    16: {"metropolis": 1.733, "glauber": 2.01},
    30: {"metropolis": 1.673, "glauber": 1.942},
    40: {"metropolis": 1.721, "glauber": 1.975},
    50: {"metropolis": 1.688, "glauber": 1.947},
}


def _us_per_step(n: int) -> Dict[str, float]:
    """Step rates for the measured size nearest to n."""
    nearest = min(US_PER_STEP_BY_N, key=lambda k: abs(k - n))
    return US_PER_STEP_BY_N[nearest]


def _fep_us_per_step(n: int) -> Dict[str, float]:
    """FEP step rates for the measured size nearest to n."""
    nearest = min(FEP_US_PER_STEP_BY_N, key=lambda k: abs(k - n))
    return FEP_US_PER_STEP_BY_N[nearest]


def _log_z_walltime_figure(graph_id: str, n: int,
                           log_z_mcmc_for_graph: Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]],
                           log_z_fep_for_graph: Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]],
                           log_z_js_sweep_for_graph: Dict[float, Dict[float, Dict[int, List[Dict]]]],
                           log_z_final_for_graph: Dict[float, Dict[float, Dict[str, float]]],
                           log_z_taylor_for_graph: Dict[float, Dict[float, Dict[str, List[Dict]]]],
                           betas: List[float], h_values: List[float],
                           inits: List[str], dyns: List[str],
                           show_rel: bool) -> Tuple[go.Figure, bool]:
    """All estimators on a fair wall-time x-axis: MCMC thermo lines, MCMC FEP
    lines (markers), FPRAS markers, Taylor markers.  MCMC/FEP/JS step counts
    are converted to seconds via microbenched per-step rates; Taylor uses its
    measured runtime.  show_rel=False -> raw log Ẑ; show_rel=True -> relative
    error vs the reference (exact log Z for n<=EXACT_MAX_N, FEP otherwise)."""
    has_exact = show_rel
    ref_name = "exact log Z" if n <= EXACT_MAX_N else "FEP reference"
    rates = _us_per_step(n)
    fep_rates = _fep_us_per_step(n)
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
        h_js_sweep = log_z_js_sweep_for_graph.get(h, {})
        h_finals = log_z_final_for_graph.get(h, {})
        h_taylor = log_z_taylor_for_graph.get(h, {})
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
                    us_per_step = rates.get(dyn, 2.0)
                    x_s = (steps * n_chains_for_target * us_per_step / 1e6)
                    group = f"beta={beta} init={init} dyn={dyn}"
                    fig.add_trace(
                        go.Scatter(
                            x=x_s, y=y, mode="lines",
                            name=f"MCMC β={beta}, {INIT_LABEL[init]}, {dyn}",
                            legendgroup=group, showlegend=is_first,
                            line=dict(color=color, width=1.4,
                                      dash=STYLE_PLOTLY[(init, dyn)]),
                            hovertemplate=(
                                f"<b>MCMC thermo β={beta}, h={h}</b><br>"
                                f"init={INIT_LABEL[init]}, dynamics={dyn}<br>"
                                f"n_chains used = {n_chains_for_target}<br>"
                                "wall time = %{x:.3g}s<br>"
                                "y = %{y:.4g}<extra></extra>"
                            ),
                        ),
                        row=row, col=col,
                    )

            # MCMC FEP lines (markers to distinguish from the thermo lines).
            # Total chain work telescopes through (b_idx+1) ladder segments,
            # each FEP_BURNIN + (sample count); converted to seconds at the
            # microbenched FEP per-step rate for the dynamics.
            beta_fep = log_z_fep_for_graph.get(h, {}).get(beta, {})
            for init in inits:
                for dyn in dyns:
                    if (init, dyn) not in beta_fep:
                        continue
                    steps, lz = beta_fep[(init, dyn)]
                    if has_exact:
                        y = np.maximum(np.abs(lz - exact_lz) / denom, 1e-10)
                    else:
                        y = lz
                    total_steps = n_chains_for_target * (FEP_BURNIN + steps)
                    x_s = total_steps * fep_rates.get(dyn, 1.8) / 1e6
                    group = f"beta={beta} init={init} dyn={dyn}"
                    fig.add_trace(
                        go.Scatter(
                            x=x_s, y=y, mode="lines+markers",
                            name=f"FEP β={beta}, {INIT_LABEL[init]}, {dyn}",
                            legendgroup=group, showlegend=is_first,
                            line=dict(color=color, width=1.0,
                                      dash=STYLE_PLOTLY[(init, dyn)]),
                            marker=dict(color=color, size=5, symbol="diamond",
                                        line=dict(color="white", width=0.5)),
                            hovertemplate=(
                                f"<b>FEP β={beta}, h={h}</b><br>"
                                f"init={INIT_LABEL[init]}, dynamics={dyn}<br>"
                                f"segments = {n_chains_for_target}<br>"
                                "wall time = %{x:.3g}s<br>"
                                "y = %{y:.4g}<extra></extra>"
                            ),
                        ),
                        row=row, col=col,
                    )

            # FPRAS markers on the wall-time x-axis.
            beta_js_sweep = h_js_sweep.get(beta, {})
            for step_n_mult in sorted(beta_js_sweep.keys()):
                recs = beta_js_sweep[step_n_mult]
                xs, ys, hovers = [], [], []
                for r in recs:
                    lz = r["log_Z"]
                    if math.isnan(lz):
                        continue
                    if has_exact:
                        yv = max(abs(lz - exact_lz) / denom, 1e-10)
                    else:
                        yv = lz
                    x_s = r["total_steps"] * rates["js"] / 1e6
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

            # LSS Taylor markers: each m is one independent computation; x is
            # its *measured* runtime_s (no µs/step estimation), y the same
            # relative error as the FPRAS markers.
            recs_by_method = h_taylor.get(beta, {})
            for method, recs in recs_by_method.items():
                xs, ys, hovers = [], [], []
                for r in recs:
                    lz = r["log_Z"]
                    if math.isnan(lz):
                        continue
                    if has_exact:
                        yv = max(abs(lz - exact_lz) / denom, 1e-10)
                    else:
                        yv = lz
                    xs.append(max(r["runtime_s"], 1e-4))
                    ys.append(yv)
                    hovers.append(
                        f"<b>Taylor β={beta}, h={h}</b><br>"
                        f"m = {r['m']}<br>"
                        f"method = {method}<br>"
                        f"runtime = {r['runtime_s']:.3f} s (measured)<br>"
                        f"log Ẑ = {lz:.4f}"
                    )
                if not xs:
                    continue
                group = f"beta={beta} src=lss"
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines+markers",
                        name=f"Taylor β={beta}",
                        legendgroup=group, showlegend=is_first,
                        line=dict(color=color, width=0.9, dash="dot"),
                        marker=dict(color=color, size=10, symbol="star",
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
            ("relative error vs " + ref_name if has_exact
             else "raw log Ẑ")
            + f" vs estimated wall time (n={n} rates) — MCMC thermo "
            f"({rates['metropolis']:.2f}/{rates['glauber']:.2f} µs/step "
            f"Metro/Glauber); JS FPRAS ({rates['js']:.2f} µs/step); "
            "Taylor uses measured runtime"),
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


def _log_z_taylor_figure(graph_id: str, n: int,
                      log_z_taylor_for_graph: Dict[float, Dict[float, Dict[str, List[Dict]]]],
                      betas: List[float], h_values: List[float],
                      show_rel: bool) -> Tuple[go.Figure, bool]:
    """LSS / Barvinok Taylor truncation: one curve per (β, method); x = m.
    show_rel=False -> raw log Ẑ vs m; show_rel=True -> relative error
    |log Ẑ − log Z| / |log Z| (log-y) vs the reference (exact log Z for
    n<=EXACT_MAX_N, the FEP estimate for larger n; r['rel_err'] is filled
    against whichever applies in main()).

    For h = 0 the LSS chain runs at h_eff = 1/n (JS field-anneal), so the
    relative error at h=0 also carries the field-anneal bias."""
    has_exact = show_rel
    ref_name = "exact log Z" if n <= EXACT_MAX_N else "FEP reference"
    ncols = 3
    nrows = math.ceil(len(h_values) / ncols)
    titles = [f"h = {h}" for h in h_values] + [""] * (nrows * ncols - len(h_values))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        shared_xaxes=True,
                        horizontal_spacing=0.07, vertical_spacing=0.14)
    for h_idx, h in enumerate(h_values):
        row, col = h_idx // ncols + 1, h_idx % ncols + 1
        is_first = (h_idx == 0)
        h_data = log_z_taylor_for_graph.get(h, {})
        for b_idx, beta in enumerate(betas):
            recs_by_method = h_data.get(beta, {})
            if not recs_by_method:
                continue
            color = PLOTLY_COLORS[b_idx % len(PLOTLY_COLORS)]
            # Each graph carries exactly one method (naive for n<=20,
            # insects for n>20); iterate whatever's present.
            for method, recs in recs_by_method.items():
                xs, ys, hovers = [], [], []
                for r in recs:
                    lz = r["log_Z"]
                    if math.isnan(lz):
                        continue
                    if has_exact:
                        rel = r["rel_err"]
                        if math.isnan(rel) or rel <= 0:
                            continue
                        y = max(rel, 1e-15)
                    else:
                        y = lz
                    xs.append(r["m"])
                    ys.append(y)
                    hovers.append(
                        f"<b>β={beta}, h={h}</b><br>"
                        f"m = {r['m']}<br>"
                        f"log Ẑ = {lz:.4f}<br>"
                        f"log Z exact (h={h}) = {r['log_Z_exact']:.4f}<br>"
                        f"|rel err| = {r['rel_err']:.2e}<br>"
                        f"truncation bound (Lemma 2.1) ≤ {r['bound']:.2e}<br>"
                        f"method = {method}, runtime = {r['runtime_s']:.2f} s"
                    )
                if not xs:
                    continue
                group = f"beta={beta}"
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines+markers", name=f"β={beta}",
                        legendgroup=group, showlegend=is_first,
                        line=dict(color=color, width=1.6),
                        marker=dict(color=color, size=7,
                                    line=dict(color="white", width=1)),
                        hovertext=hovers, hoverinfo="text",
                    ),
                    row=row, col=col,
                )
        fig.update_xaxes(title_text="truncation order m", row=row, col=col)
        if has_exact:
            fig.update_yaxes(type="log",
                             title_text="|log Ẑ − log Z| / |log Z|",
                             row=row, col=col)
        else:
            fig.update_yaxes(title_text="log Ẑ", row=row, col=col)
    fig.update_layout(
        title=dict(text=(f"Taylor truncation vs order m ({graph_id}, n={n}) — "
                         + (f"relative error vs {ref_name}" if has_exact
                            else "raw log Ẑ")
                         + "; h=0 uses h_eff = 1/n (JS field-anneal)"),
                   x=0.5, xanchor="center", font=dict(size=11)),
        height=320 * nrows + 120,
        hovermode="closest", dragmode="pan",
        legend=dict(title="β", itemsizing="constant",
                    bgcolor="rgba(255,255,255,0.92)",
                    groupclick="togglegroup"),
        template="plotly_white",
    )
    return fig, has_exact


def _segments_table_html(graph_id: str,
                         log_z_js_sweep_for_graph: Dict[float, Dict[float, Dict[int, List[Dict]]]],
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
            recs = (log_z_js_sweep_for_graph.get(h, {})
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
<i>low-temp</i> (the all-aligned ground states +1<sup>n</sup> / −1<sup>n</sup>,
i.e. a zero-temperature start) and <i>high-temp</i> (each spin independently
+1 or −1 — the β=0, infinite-temperature start).
</p>
</details>
<details>
<summary>What are the log Z views?</summary>
<p>Several views, each with 5 h panels.</p>
<ul>
<li><b>log Z (MCMC thermo)</b> — same spin chains as the energy view,
post-processed by trapezoidal integration of ⟨E⟩ over β
(<code>log Z(β,h) = n·log 2 − ∫₀^β ⟨E⟩(β',h) dβ'</code>).  One line per
(β, init, dyn); x = per-chain steps.</li>
<li><b>log Z (MCMC FEP)</b> — same idea, but estimated by free-energy
perturbation (Zwanzig) telescoping along the β-ladder instead of
integrating ⟨E⟩:
<code>log Z(β<sub>k+1</sub>) = n·log 2 + Σ<sub>j≤k</sub> log ⟨ e<sup>−Δβ<sub>j</sub>·E</sup> ⟩<sub>β<sub>j</sub></sub></code>.
Each ratio is exact in expectation, so there is <i>no</i> trapezoidal
grid-discretization bias — validated against exact log Z at n=16, it has
~2× lower mean error than trapezoidal on the same grid, with the biggest
gains in the β≈0.5–1.8 transition region.  One line per (β, init, dyn) —
same Metropolis/Glauber × low-temp/high-temp toggles as the energy view;
x = post-burn-in samples per segment.</li>
<li><b>log Z (Jerrum-Sinclair)</b> — each marker is one independent full
FPRAS run at fixed <code>samples_per_segment ∈ {100,300,1000,3000,10000}</code>;
lines connect runs at the same β.  Dash style encodes the schedule step
(<code>1/n</code> solid, <code>1/(4n)</code> dashed, <code>1/(20n)</code> dotted).
x = total FPRAS chain steps = <code>n_segments × (burnin + samples_per_segment)</code>.
Schedule-length table below the chart.</li>
<li><b>log Z (Taylor)</b> — deterministic truncation of
<code>log Z(λ)</code> as a polynomial in the edge activity, using the
Barvinok / Patel-Regts approach.  x = truncation order m; one curve per
β.  For h=0 we run at <code>h_eff = 1/n</code> to keep the activity off
the Lee-Yang circle; the relative error then includes both the Taylor
truncation error and the field-anneal bias.</li>
</ul>
<p>Each log-Z view shows <b>two stacked figure sets</b>: raw log Ẑ on top,
then relative error below.  The relative-error reference is the brute-force
exact log Z for n ≤ 20, and the FEP estimate itself for larger n (where no
exact is tractable).</p>
</details>
<details>
<summary>What is the "log Z vs wall time" view?</summary>
<p>All estimators replotted with x = wall time (seconds), so you can fairly
compare which one reaches a target accuracy faster.  Step rates were
microbenched on this machine (one number per size, roughly flat in n on
3-regular graphs):
MCMC-thermo Metropolis ≈ 1.4–1.5 / Glauber ≈ 1.3–1.4 µs/step;
MCMC-FEP (adds an online log-sum-exp per step) Metropolis ≈ 1.67–1.73 /
Glauber ≈ 1.94–2.0 µs/step; JS subgraphs ≈ 0.55 µs/step;
Taylor markers use the measured per-run wall time directly.  FEP markers
(diamonds) account for the full telescoping cost: its total chain steps are
<code>(#segments) × (burn-in + samples)</code>, summed over the ladder
segments below the target β.</p>
</details>
<details>
<summary>How does the Jerrum-Sinclair FPRAS work? (telescoping)</summary>
<p>JS 1990 maps the Ising partition function to a sum over <i>edge subsets</i>
X ⊆ E, weighted by the number of odd-degree vertices <code>|odd(X)|</code>:
<code>Z(β, h) = const · Σ<sub>X⊆E</sub> λ<sup>|X|</sup> · μ<sup>|odd(X)|</sup></code>
with edge activity <code>λ = tanh β</code> and field activity
<code>μ = tanh(β·h)</code>. The "subgraphs-world" Markov chain proposes a
random edge flip and accepts with Metropolis probability — fast to mix
because the state space is structured.</p>
<p>The hard part is the normalising constant Z.  JS gets it by
<b>telescoping in μ</b>, annealing from the <i>easy</i> end
<code>μ = 1</code> <b>down</b> to the target:</p>
<ul>
<li>At <code>μ = 1</code> every subset contributes equally in μ, so
<code>Z'(1) = Σ<sub>X⊆E</sub> λ<sup>|X|</sup> = Π<sub>e</sub>(1 + λ<sub>e</sub>)</code>
— a <b>closed form</b>, no MCMC.  (The opposite end <code>μ = 0</code> keeps
only fully-even subgraphs and <i>is</i> the hard h = 0 partition function —
so we anneal away from it, not toward a closed form at 0.)</li>
<li>Pick a <b>decreasing</b> schedule
<code>1 = μ<sub>0</sub> &gt; μ<sub>1</sub> &gt; ··· &gt; μ<sub>K</sub> = tanh(β·h)</code>,
the paper's step being <code>μ<sub>k</sub> = (n−k)/n</code> (K ≈ n).</li>
<li>Telescope: <code>log Z(μ<sub>K</sub>) = log Z(μ<sub>0</sub>=1) +
Σ<sub>k</sub> log [Z(μ<sub>k+1</sub>) / Z(μ<sub>k</sub>)]</code>.</li>
<li>Each ratio is an expectation under
<code>π<sub>k</sub> ∝ λ<sup>|X|</sup> μ<sub>k</sub><sup>|odd(X)|</sup></code>:
<code>Z(μ<sub>k+1</sub>)/Z(μ<sub>k</sub>) = E<sub>X∼π<sub>k</sub></sub>
[(μ<sub>k+1</sub>/μ<sub>k</sub>)<sup>|odd(X)|</sup>]</code>.  The denominator
<code>μ<sub>k</sub></code> is never 0 (it starts at 1); only the final
target can be 0 (when h = 0), where the ratio becomes
<code>E[0<sup>|odd(X)|</sup>]</code> = the fraction of <i>even</i> subgraphs
sampled at <code>μ<sub>K−1</sub></code>.</li>
<li>Estimate each expectation by running the chain at <code>μ<sub>k</sub></code>
and averaging the integrand over <code>samples_per_segment</code> states.</li>
</ul>
<p>The <code>step_n_mult</code> knob refines the schedule from K=n to
K=4n or K=20n; finer schedules → smaller per-link variance but more
links to estimate.  The sweep view shows the variance/cost tradeoff
directly.</p>
</details>
<details>
<summary>How does the Barvinok/Taylor truncation work?</summary>
<p>Write the Ising partition function as a polynomial in the edge
activity:
<code>Z(λ) = Σ<sub>k</sub> c<sub>k</sub> λ<sup>k</sup></code>,
where <code>c<sub>k</sub></code> counts size-k subsets of vertices
weighted by <code>β<sup>|cut(S)|</sup></code> (or, dually, depends only
on connected induced subgraphs of size ≤ k).  Then
<code>log Z(λ) = Σ<sub>k</sub> f<sup>(k)</sup>(0) · λ<sup>k</sup> / k!</code>
is an analytic function whose derivatives at 0 are recovered from
<code>c<sub>1</sub>, ..., c<sub>k</sub></code> by Newton's identities.</p>
<p>The <b>Barvinok / Patel-Regts FPTAS</b> truncates this Taylor series
at order m:
<code>log Ẑ<sub>m</sub>(λ) = Σ<sub>k≤m</sub> f<sup>(k)</sup>(0) λ<sup>k</sup>/k!</code>,
evaluated at the activity <code>λ = exp(2β·h<sub>eff</sub>)</code>.
By the <i>Lee-Yang theorem</i> all zeros of <code>Z(λ)</code> for
ferromagnetic Ising lie on the unit circle <code>|λ| = 1</code>, so
inside that disk <code>log Z</code> is analytic.  Writing
<code>a = min(|λ|, 1/|λ|) &lt; 1</code> (the <code>λ ↔ 1/λ</code>
symmetry handles <code>|λ| &gt; 1</code>), the order-m truncation error
is bounded (Lemma 2.1) by
<code>n·a<sup>m+1</sup> / ((m+1)(1 − a))</code> and decays like
<code>a<sup>m</sup></code> — <i>polynomial time</i> for any fixed
accuracy whenever <code>a</code> is bounded below 1, even though Ising
itself is #P-hard.</p>
<p>Two ways to extract the coefficients:</p>
<ul>
<li><b>naive</b> — enumerate every size-k subset, sum its weight.  Cost
<code>C(n, k) · |E|</code> per <code>c<sub>k</sub></code>; fine for
n ≲ 20.</li>
<li><b>insects</b> (Patel-Regts §3.4) — a DP over connected induced
subgraphs of size ≤ m.  For bounded-degree graphs the number of such
subgraphs is poly(n) for fixed m, so the whole truncation is poly(n).</li>
</ul>
<p>For <code>h = 0</code> the activity <code>λ = exp(2βh) = 1</code>
sits exactly on the Lee-Yang circle, so <code>a = 1</code> and the bound
diverges.  We use the Jerrum-Sinclair field-anneal trick — replace h with
<code>h<sub>eff</sub> = 1/n</code> — to push λ off the circle
(<code>a = exp(−2β/n) &lt; 1</code>); the extra bias goes to 0 as
n → ∞.</p>
</details>
</p>
"""

CONTROLS_HTML_TEMPLATE = """
<div class="controls">
  <fieldset id="ctl-view">
    <legend>View</legend>
    <label><input type="radio" name="viewsel" checked value="energy"> energy convergence</label>
    <label><input type="radio" name="viewsel" value="logz_mcmc"> log Z (MCMC thermo)</label>
    <label><input type="radio" name="viewsel" value="logz_fep"> log Z (MCMC FEP)</label>
    <label><input type="radio" name="viewsel" value="logz_js_sweep"> log Z (Jerrum-Sinclair)</label>
    <label><input type="radio" name="viewsel" value="logz_taylor"> log Z (Taylor)</label>
    <label><input type="radio" name="viewsel" value="logz_walltime"> log Z vs wall time</label>
  </fieldset>
  <fieldset id="ctl-graph">
    <legend>Graph</legend>
    {graph_radios}
  </fieldset>
  <fieldset id="ctl-init" class="view-only-energy">
    <legend>Initial distribution</legend>
    <label><input type="checkbox" data-filter="init" data-value="ground"> low-temp</label>
    <label><input type="checkbox" checked data-filter="init" data-value="uniform"> high-temp</label>
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
  const VIEWS_WITH_INIT_DYN = new Set(['energy', 'logz_mcmc', 'logz_fep', 'logz_walltime']);
  const PLOT_PREFIXES = ['convfig-', 'logzmcmcfig-', 'logzfepfig-', 'logzfig-', 'logzwalltimefig-', 'logztaylorfig-'];

  function ensureRendered(div) {
    if (!div || div.data) return;  // already rendered
    const script = document.querySelector('script[data-figdiv="' + div.id + '"]');
    if (!script) return;
    const spec = JSON.parse(script.textContent);
    Plotly.newPlot(div, spec.data, spec.layout, spec.config || {responsive: true});
    div.classList.add('plotly-rendered');
  }
  function renderVisibleNow() {
    document.querySelectorAll('.graph-section:not(.hidden) .plotly-lazy').forEach(div => {
      // Skip lazy divs inside a view that is currently hidden.
      const viewWrap = div.closest('[data-view]');
      if (viewWrap && viewWrap.classList.contains('view-hidden')) return;
      ensureRendered(div);
    });
  }
  function applyTraceFilters() {
    const initSet = new Set(Array.from(document.querySelectorAll('[data-filter="init"]:checked')).map(e => e.dataset.value));
    const dynSet = new Set(Array.from(document.querySelectorAll('[data-filter="dyn"]:checked')).map(e => e.dataset.value));
    const betaSet = new Set(Array.from(document.querySelectorAll('[data-filter="beta"]:checked')).map(e => e.dataset.value));
    const sel = document.querySelector('input[name="graphsel"]:checked');
    if (!sel) return;
    // Each view may hold more than one figure with the same prefix (e.g. the
    // stacked raw-log-Z and relative-error sets), so filter all of them.
    PLOT_PREFIXES.forEach(prefix => {
      document.querySelectorAll('.graph-section:not(.hidden) [id^="' + prefix + '"]').forEach(div => {
        if (!div.data) return;
        const visibility = div.data.map(trace => {
          const m = parseGroup(trace.legendgroup);
          if (!m) return true;
          if (m.init !== undefined && !initSet.has(m.init)) return false;
          if (m.dyn !== undefined && !dynSet.has(m.dyn)) return false;
          if (m.beta !== undefined && !betaSet.has(m.beta)) return false;
          if (m.method !== undefined && !methodSet.has(m.method)) return false;
          return true;
        });
        Plotly.restyle(div, {visible: visibility});
      });
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
      renderVisibleNow();
      applyTraceFilters();
      PLOT_PREFIXES.forEach(prefix => {
        document.querySelectorAll('.graph-section:not(.hidden) [data-view]:not(.view-hidden) [id^="' + prefix + '"]').forEach(div => {
          if (window.Plotly && div.layout) Plotly.Plots.resize(div);
        });
      });
    }, 30);
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
      renderVisibleNow();
      ['graphfig-'].concat(PLOT_PREFIXES).forEach(prefix => {
        document.querySelectorAll('.graph-section:not(.hidden) [id^="' + prefix + '"]').forEach(div => {
          if (window.Plotly && div.layout) Plotly.Plots.resize(div);
        });
      });
      applyTraceFilters();
      applyView();
    }, 30);
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
                  log_z_js_sweep_for_graph: Dict[float, Dict[float, Dict[int, List[Dict]]]],
                  log_z_mcmc_for_graph: Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]],
                  log_z_fep_for_graph: Dict[float, Dict[float, Tuple[np.ndarray, np.ndarray]]],
                  log_z_taylor_for_graph: Dict[float, Dict[float, Dict[str, List[Dict]]]],
                  betas: List[float], h_values: List[float],
                  inits: List[str], dyns: List[str],
                  is_default_visible: bool) -> str:
    graph_fig = _graph_figure(G_nx, title=f"{graph_id} — spring layout")
    conv_fig, _ = _convergence_figure(graph_id, graph_data, exact,
                                      betas, h_values, inits, dyns)
    seg_table_html = _segments_table_html(graph_id, log_z_js_sweep_for_graph,
                                          betas, h_values)

    # Each log-Z view shows two stacked figure sets: raw log Ẑ on top, then
    # relative error below (vs exact for n<=EXACT_MAX_N, vs the FEP estimate
    # otherwise).  abs/rel ids share the view's prefix so the lazy-render and
    # trace-filter JS (which key on the prefix) pick up both.
    def _two_set(view: str, prefix: str, make, extra: str = "") -> str:
        abs_html = _fig_lazy_html(make(False), f"{prefix}-abs-{graph_id}")
        rel_html = _fig_lazy_html(make(True), f"{prefix}-rel-{graph_id}")
        return (f'<div class="{prefix}" data-view="{view}">'
                f'{abs_html}{rel_html}{extra}</div>')

    graph_html = _fig_lazy_html(graph_fig, f"graphfig-{graph_id}")
    conv_html = _fig_lazy_html(conv_fig, f"convfig-{graph_id}")
    graph_wrap = f'<div class="graphfig">{graph_html}</div>'
    conv_wrap = f'<div class="convfig" data-view="energy">{conv_html}</div>'

    mcmc_wrap = _two_set(
        "logz_mcmc", "logzmcmcfig",
        lambda sr: _log_z_mcmc_figure(
            graph_id, n, log_z_mcmc_for_graph, log_z_for_graph,
            betas, h_values, inits, dyns, show_rel=sr)[0])
    fep_wrap = _two_set(
        "logz_fep", "logzfepfig",
        lambda sr: _log_z_fep_figure(
            graph_id, n, log_z_fep_for_graph, log_z_for_graph,
            betas, h_values, inits, dyns, show_rel=sr))
    js_wrap = _two_set(
        "logz_js_sweep", "logzfig",
        lambda sr: _log_z_figure(
            graph_id, n, log_z_js_sweep_for_graph, log_z_for_graph,
            betas, h_values, show_rel=sr)[0],
        extra=seg_table_html)
    taylor_wrap = _two_set(
        "logz_taylor", "logztaylorfig",
        lambda sr: _log_z_taylor_figure(
            graph_id, n, log_z_taylor_for_graph, betas, h_values,
            show_rel=sr)[0])
    walltime_wrap = _two_set(
        "logz_walltime", "logzwalltimefig",
        lambda sr: _log_z_walltime_figure(
            graph_id, n, log_z_mcmc_for_graph, log_z_fep_for_graph,
            log_z_js_sweep_for_graph, log_z_for_graph, log_z_taylor_for_graph,
            betas, h_values, inits, dyns, show_rel=sr)[0])

    section_class = "graph-section" + ("" if is_default_visible else " hidden")
    return (
        f'<section class="{section_class}" data-graph="{graph_id}">'
        f'<h2>{graph_id}  (3-regular, n={n})</h2>'
        + graph_wrap + conv_wrap + mcmc_wrap + fep_wrap
        + js_wrap + taylor_wrap + walltime_wrap +
        '</section>'
    )


def render_combined_html(out_path: str,
                         data: Dict, graph_ids: List[str],
                         exact: Dict[Tuple[str, float, float], float],
                         log_z: Dict[str, Dict[float, Dict[float, Dict[str, float]]]],
                         log_z_js_sweep: Dict[str, Dict[float, Dict[float, Dict[int, List[Dict]]]]],
                         log_z_mcmc: Dict[str, Dict[float, Dict[float, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]]],
                         log_z_fep: Dict[str, Dict[float, Dict[float, Tuple[np.ndarray, np.ndarray]]]],
                         log_z_taylor: Dict[str, Dict[float, Dict[float, Dict[str, List[Dict]]]]],
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
            log_z.get(gid, {}), log_z_js_sweep.get(gid, {}),
            log_z_mcmc.get(gid, {}), log_z_fep.get(gid, {}),
            log_z_taylor.get(gid, {}),
            betas, h_values, inits, dyns,
            is_default_visible=(gid == default_selected),
        ))
    # Use the local plotly.js bundle that ships next to this script (no
    # internet at view time).  Fetched once into vendor/ on first run.
    _ensure_plotly_js()
    plotly_cdn = '<script src="vendor/plotly.min.js"></script>'
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
    log_z_js_sweep = load_log_z_js_sweep(LOG_Z_JS_SWEEP_CSV)
    log_z_mcmc = load_log_z_mcmc(LOG_Z_MCMC_CSV)
    log_z_fep = load_log_z_fep(LOG_Z_FEP_CSV)
    log_z_taylor = load_log_z_taylor(LOG_Z_TAYLOR_CSV)

    # Reference for log-Z relative error: brute-force exact where we have it
    # (n <= EXACT_MAX_N), else the FEP telescoping estimate (no trapezoidal
    # grid bias; validated vs exact at n=16).  The earlier long-Glauber
    # thermo-integrated reference is NOT used -- it drifted ~18% at high beta.
    FEP_REF_KEY = ("uniform", "glauber")     # high-temp Glauber FEP
    for gid, h_map in log_z_fep.items():
        for h, beta_map in h_map.items():
            for beta, series in beta_map.items():
                sl = series.get(FEP_REF_KEY) or next(iter(series.values()))
                fep_final = float(sl[1][-1])
                (log_z.setdefault(gid, {}).setdefault(h, {})
                    .setdefault(beta, {}).setdefault("exact", fep_final))
    # Fill Taylor relative error against the (now-present) reference where the
    # producer left it NaN (no brute-force exact at n > EXACT_MAX_N).
    for gid, h_map in log_z_taylor.items():
        for h, beta_map in h_map.items():
            for beta, method_map in beta_map.items():
                ref = log_z.get(gid, {}).get(h, {}).get(beta, {}).get("exact")
                if ref is None:
                    continue
                denom = abs(ref) if abs(ref) > 1e-12 else 1.0
                for recs in method_map.values():
                    for r in recs:
                        if not math.isnan(r["rel_err"]):
                            continue
                        r["log_Z_exact"] = ref
                        r["rel_err"] = abs(r["log_Z"] - ref) / denom

    print(f"loaded {len(graph_ids)} graphs: {graph_ids}")
    print(f"  betas={betas}\n  h={h_values}\n  inits={inits}\n  dyns={dyns}")
    print(f"  log_z graphs:        {sorted(log_z.keys())}")
    print(f"  log_z js sweep:      {sorted(log_z_js_sweep.keys())}")
    print(f"  log_z mcmc graphs:   {sorted(log_z_mcmc.keys())}")
    print(f"  log_z fep graphs:    {sorted(log_z_fep.keys())}")
    print(f"  log_z taylor:        {sorted(log_z_taylor.keys())}")

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
                         log_z_js_sweep, log_z_mcmc, log_z_fep, log_z_taylor,
                         betas, h_values, inits, dyns)
    print(f"  wrote {COMBINED_HTML}")


if __name__ == "__main__":
    main()
