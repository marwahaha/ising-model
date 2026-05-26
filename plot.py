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
</p>
"""

CONTROLS_HTML_TEMPLATE = """
<div class="controls">
  <fieldset id="ctl-graph">
    <legend>Graph</legend>
    {graph_radios}
  </fieldset>
  <fieldset id="ctl-init">
    <legend>Initial distribution</legend>
    <label><input type="checkbox" checked data-filter="init" data-value="ground"> ground</label>
    <label><input type="checkbox" checked data-filter="init" data-value="uniform"> uniform</label>
    <div class="shortcut-btns">
      <button type="button" onclick="setAll('init', true)">all</button>
      <button type="button" onclick="setAll('init', false)">none</button>
    </div>
  </fieldset>
  <fieldset id="ctl-dyn">
    <legend>Dynamics</legend>
    <label><input type="checkbox" checked data-filter="dyn" data-value="metropolis"> metropolis</label>
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
  // legendgroup is encoded in Python as "beta=<x> init=<y> dyn=<z>".
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
  function applyTraceFilters() {
    // Only restyle the currently visible chart — restyling all 6 on every
    // checkbox click is expensive (200 traces each).  When the user switches
    // graphs we re-apply filters then.
    const initSet = new Set(Array.from(document.querySelectorAll('[data-filter="init"]:checked')).map(e => e.dataset.value));
    const dynSet = new Set(Array.from(document.querySelectorAll('[data-filter="dyn"]:checked')).map(e => e.dataset.value));
    const betaSet = new Set(Array.from(document.querySelectorAll('[data-filter="beta"]:checked')).map(e => e.dataset.value));
    const sel = document.querySelector('input[name="graphsel"]:checked');
    if (!sel) return;
    const div = document.getElementById('convfig-' + sel.value);
    if (!div || !div.data) return;
    const visibility = div.data.map(trace => {
      const m = parseGroup(trace.legendgroup);
      if (!m) return true;
      if (!initSet.has(m.init)) return false;
      if (!dynSet.has(m.dyn)) return false;
      if (!betaSet.has(m.beta)) return false;
      return true;
    });
    Plotly.restyle(div, {visible: visibility});
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
    // Plotly sometimes needs a resize after a previously hidden chart appears,
    // and we need to re-apply the current filter state to the now-visible
    // chart (we only restyle the visible one on filter changes).
    setTimeout(() => {
      document.querySelectorAll('.graph-section:not(.hidden) [id^="convfig-"], .graph-section:not(.hidden) [id^="graphfig-"]').forEach(div => {
        if (window.Plotly && div.layout) Plotly.Plots.resize(div);
      });
      applyTraceFilters();
    }, 50);
  }
  document.querySelectorAll('.controls input[type=checkbox]').forEach(cb => {
    cb.addEventListener('change', applyTraceFilters);
  });
  document.querySelectorAll('input[name="graphsel"]').forEach(r => {
    r.addEventListener('change', applyGraphSelection);
  });
  function setAll(filter, value) {
    document.querySelectorAll('[data-filter="' + filter + '"]').forEach(cb => cb.checked = value);
    applyTraceFilters();
  }
  document.addEventListener('DOMContentLoaded', applyGraphSelection);
</script>
"""


def _make_section(graph_id: str, n: int, G_nx: nx.Graph, graph_data: Dict,
                  exact: Dict[Tuple[str, float, float], float],
                  betas: List[float], h_values: List[float],
                  inits: List[str], dyns: List[str],
                  is_default_visible: bool) -> str:
    graph_fig = _graph_figure(G_nx, title=f"{graph_id} — spring layout")
    conv_fig, has_exact = _convergence_figure(graph_id, graph_data, exact,
                                              betas, h_values, inits, dyns)
    graph_html = graph_fig.to_html(include_plotlyjs=False, full_html=False,
                                   div_id=f"graphfig-{graph_id}")
    conv_html = conv_fig.to_html(include_plotlyjs=False, full_html=False,
                                 div_id=f"convfig-{graph_id}")
    # Wrap so JS can grab .graphfig / .convfig classes for filtering.
    graph_wrap = f'<div class="graphfig">{graph_html}</div>'
    conv_wrap = f'<div class="convfig">{conv_html}</div>'
    section_class = "graph-section" + ("" if is_default_visible else " hidden")
    return (
        f'<section class="{section_class}" data-graph="{graph_id}">'
        f'<h2>{graph_id}  (3-regular, n={n})</h2>'
        + graph_wrap + conv_wrap +
        '</section>'
    )


def render_combined_html(out_path: str,
                         data: Dict, graph_ids: List[str],
                         exact: Dict[Tuple[str, float, float], float],
                         betas: List[float], h_values: List[float],
                         inits: List[str], dyns: List[str]) -> None:
    # Default-selected graph is the first one (n16_graph0 by ordering).
    default_selected = graph_ids[0] if graph_ids else None

    graph_radios = "".join(
        f'<label><input type="radio" name="graphsel"'
        f'{" checked" if gid == default_selected else ""} value="{gid}"> {gid}</label>'
        for gid in graph_ids
    )
    beta_checkboxes = "".join(
        f'<label><input type="checkbox" checked data-filter="beta" data-value="{b}"> {b}</label>'
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
            gid, n, G_nx, data[gid], exact, betas, h_values, inits, dyns,
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
    print(f"loaded {len(graph_ids)} graphs: {graph_ids}")
    print(f"  betas={betas}\n  h={h_values}\n  inits={inits}\n  dyns={dyns}")

    # Per-graph PNG (archival).
    for graph_id in graph_ids:
        G_nx = load_graph(graph_id)
        n = G_nx.number_of_nodes()
        png = f"convergence_{graph_id}.png"
        static_plot(graph_id, n, data[graph_id], exact, betas, h_values,
                    inits, dyns, png)
        print(f"  wrote {png}")

    # One combined interactive page.
    render_combined_html(COMBINED_HTML, data, graph_ids, exact,
                         betas, h_values, inits, dyns)
    print(f"  wrote {COMBINED_HTML}")


if __name__ == "__main__":
    main()
