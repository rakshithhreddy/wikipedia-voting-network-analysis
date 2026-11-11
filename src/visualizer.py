"""Network Visualization Module for Wikipedia Voting Network.

Generates interactive 2D Plotly visualizations with custom node sizing,
community coloring, physics layouts, and comprehensive hover tooltips.
Also supports Matplotlib high-resolution publication-quality figure generation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def create_plotly_network_figure(
    subgraph: nx.DiGraph,
    df_metrics: pd.DataFrame,
    partition: Optional[Dict[int, int]] = None,
    layout_type: str = "spring",
    color_by: str = "community",
    size_by: str = "in_degree",
    node_size_scale: float = 1.0,
    title: str = "Wikipedia Voting Network Subgraph",
) -> go.Figure:
    """Generate an interactive Plotly graph figure from a directed subgraph.

    Parameters
    ----------
    subgraph : nx.DiGraph
        Sampled directed subgraph.
    df_metrics : pd.DataFrame
        Centrality metrics table.
    partition : Optional[Dict[int, int]]
        Community mapping {node_id: community_id}.
    layout_type : str
        Graph layout algorithm ('spring', 'kamada_kawai', 'circular', 'spiral').
    color_by : str
        Attribute for node coloring ('community', 'in_degree', 'pagerank', 'betweenness_centrality').
    size_by : str
        Attribute for node diameter ('in_degree', 'pagerank', 'out_degree', 'total_degree').
    node_size_scale : float
        Multiplier for node sizing.
    title : str
        Title for the figure.

    Returns
    -------
    go.Figure
        Configured Plotly figure.
    """
    if subgraph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="No nodes in selected subgraph")
        return fig

    # Fast metric lookup
    metrics_indexed = df_metrics.set_index("node_id")

    # Compute layout coordinates
    if layout_type == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(subgraph)
        except Exception:
            pos = nx.spring_layout(subgraph, seed=42, k=1.0 / np.sqrt(subgraph.number_of_nodes()))
    elif layout_type == "circular":
        pos = nx.circular_layout(subgraph)
    elif layout_type == "spiral":
        pos = nx.spiral_layout(subgraph)
    else:  # spring layout
        pos = nx.spring_layout(
            subgraph,
            seed=42,
            k=1.2 / np.sqrt(subgraph.number_of_nodes()),
            iterations=60,
        )

    # Edge traces
    edge_x = []
    edge_y = []
    for u, v in subgraph.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.0, color="rgba(150, 160, 180, 0.45)"),
        hoverinfo="none",
        mode="lines",
        name="Directed Vote (u → v)",
    )

    # Node traces
    node_x = []
    node_y = []
    node_hover = []
    node_sizes = []
    node_colors = []
    node_text = []

    # Sizing normalization
    size_vals = []
    for node in subgraph.nodes():
        if node in metrics_indexed.index:
            row = metrics_indexed.loc[node]
            val = float(row.get(size_by, 1.0))
        else:
            val = 1.0
        size_vals.append(val)

    min_s, max_s = (min(size_vals), max(size_vals)) if size_vals else (1, 1)

    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

        # Retrieve metrics
        if node in metrics_indexed.index:
            row = metrics_indexed.loc[node]
            in_deg = int(row.get("in_degree", 0))
            out_deg = int(row.get("out_degree", 0))
            pr = float(row.get("pagerank", 0.0))
            bt = float(row.get("betweenness_centrality", 0.0))
        else:
            in_deg = subgraph.in_degree(node)
            out_deg = subgraph.out_degree(node)
            pr = 0.0
            bt = 0.0

        comm_id = partition.get(node, "N/A") if partition else "N/A"

        # Sizing
        if max_s > min_s:
            raw_s = float(
                metrics_indexed.loc[node].get(size_by, 1.0)
                if node in metrics_indexed.index
                else 1.0
            )
            # Map to range [10, 36]
            norm_s = 12 + 28 * ((raw_s - min_s) / (max_s - min_s))
        else:
            norm_s = 16
        node_sizes.append(norm_s * node_size_scale)

        # Coloring
        if color_by == "community" and partition:
            node_colors.append(comm_id)
        elif color_by in metrics_indexed.columns and node in metrics_indexed.index:
            node_colors.append(float(metrics_indexed.loc[node].get(color_by, 0.0)))
        else:
            node_colors.append(in_deg)

        # Hover text with rich formatting
        hover_str = (
            f"<b>User Node:</b> {node}<br>"
            f"<b>Community:</b> {comm_id}<br>"
            f"<b>Votes Received (In-Degree):</b> {in_deg:,}<br>"
            f"<b>Votes Cast (Out-Degree):</b> {out_deg:,}<br>"
            f"<b>PageRank:</b> {pr:.6f}<br>"
            f"<b>Betweenness Centrality:</b> {bt:.6f}"
        )
        node_hover.append(hover_str)

    # Determine colorscale and title
    if color_by == "community":
        colorscale = "Rainbow"
        cbar_title = "Community ID"
    elif color_by == "pagerank":
        colorscale = "Viridis"
        cbar_title = "PageRank"
    elif color_by == "betweenness_centrality":
        colorscale = "Plasma"
        cbar_title = "Betweenness"
    else:
        colorscale = "Turbo"
        cbar_title = "In-Degree"

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text if subgraph.number_of_nodes() <= 60 else None,
        textposition="top center",
        textfont=dict(size=9, color="#2c3e50"),
        hoverinfo="text",
        hovertext=node_hover,
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=14,
                title=dict(text=cbar_title, font=dict(size=12)),
                xanchor="left",
                len=0.75,
            ),
            line=dict(width=1.5, color="#ffffff"),
        ),
        name="Wikipedia Users",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16, color="#1e293b")),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(248, 250, 252, 0.95)",
            paper_bgcolor="rgba(255, 255, 255, 1.0)",
            height=620,
        ),
    )

    return fig


def plot_static_degree_histogram(
    in_degrees: List[int],
    out_degrees: List[int],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate high-resolution Matplotlib dual log-log degree distribution plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # In-degree log-log
    in_counts = pd.Series(in_degrees).value_counts().sort_index()
    in_counts = in_counts[in_counts.index > 0]
    ax1.scatter(in_counts.index, in_counts.values, alpha=0.7, color="#2563eb", edgecolors="none")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("In-Degree Distribution (Votes Received) [Log-Log]")
    ax1.set_xlabel("In-Degree (k_in)")
    ax1.set_ylabel("Frequency P(k_in)")
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    # Out-degree log-log
    out_counts = pd.Series(out_degrees).value_counts().sort_index()
    out_counts = out_counts[out_counts.index > 0]
    ax2.scatter(out_counts.index, out_counts.values, alpha=0.7, color="#16a34a", edgecolors="none")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Out-Degree Distribution (Votes Cast) [Log-Log]")
    ax2.set_xlabel("Out-Degree (k_out)")
    ax2.set_ylabel("Frequency P(k_out)")
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved degree distribution plot to {save_path}")

    return fig
