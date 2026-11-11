"""Network Metrics Module for Wikipedia Voting Network.

Computes macro-level Social Network Analysis (SNA) metrics including:
- Graph density
- Reciprocity
- Degree statistics and distributions
- Weakly and Strongly Connected Components
- Clustering coefficients
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_global_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """Compute comprehensive global macro-level SNA metrics for a directed graph.

    Parameters
    ----------
    G : nx.DiGraph
        The directed Wikipedia voting graph.

    Returns
    -------
    Dict[str, Any]
        Dictionary of formatted metrics, numerical values, and interpretations.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return {"error": "Graph is empty"}

    # Directed density: E / (N * (N - 1))
    possible_edges = n_nodes * (n_nodes - 1)
    density = n_edges / possible_edges if possible_edges > 0 else 0.0

    # Reciprocity: proportion of edges that are bidirectional
    reciprocity_val = nx.reciprocity(G)

    # In and Out degrees
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    total_degrees = [in_d + out_d for in_d, out_d in zip(in_degrees, out_degrees)]

    avg_in_degree = float(np.mean(in_degrees))
    avg_out_degree = float(np.mean(out_degrees))
    median_in_degree = float(np.median(in_degrees))
    median_out_degree = float(np.median(out_degrees))

    max_in_degree = int(np.max(in_degrees))
    max_out_degree = int(np.max(out_degrees))

    # Connected Components
    wcc = list(nx.weakly_connected_components(G))
    num_wcc = len(wcc)
    largest_wcc_size = len(max(wcc, key=len)) if wcc else 0
    wcc_coverage = (largest_wcc_size / n_nodes) * 100 if n_nodes > 0 else 0

    scc = list(nx.strongly_connected_components(G))
    num_scc = len(scc)
    largest_scc_size = len(max(scc, key=len)) if scc else 0
    scc_coverage = (largest_scc_size / n_nodes) * 100 if n_nodes > 0 else 0

    # Undirected projection clustering coefficient
    # Transitivity / clustering measures local triadic closure
    G_undirected = G.to_undirected()
    avg_clustering = nx.average_clustering(G_undirected)
    transitivity = nx.transitivity(G_undirected)

    # Isolated nodes or zero-degree nodes
    zero_in = sum(1 for d in in_degrees if d == 0)
    zero_out = sum(1 for d in out_degrees if d == 0)

    metrics = {
        "node_count": n_nodes,
        "edge_count": n_edges,
        "density": density,
        "reciprocity": reciprocity_val,
        "avg_in_degree": avg_in_degree,
        "avg_out_degree": avg_out_degree,
        "median_in_degree": median_in_degree,
        "median_out_degree": median_out_degree,
        "max_in_degree": max_in_degree,
        "max_out_degree": max_out_degree,
        "num_wcc": num_wcc,
        "largest_wcc_size": largest_wcc_size,
        "wcc_coverage_pct": wcc_coverage,
        "num_scc": num_scc,
        "largest_scc_size": largest_scc_size,
        "scc_coverage_pct": scc_coverage,
        "undirected_avg_clustering": avg_clustering,
        "undirected_transitivity": transitivity,
        "pure_voters_count": zero_in,  # cast votes but never received any
        "pure_candidates_count": zero_out,  # received votes but never voted
    }

    return metrics


def get_degree_distribution_data(G: nx.DiGraph) -> pd.DataFrame:
    """Calculate the frequency distribution of in-degrees, out-degrees, and total degrees.

    Parameters
    ----------
    G : nx.DiGraph
        Directed voting graph.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by degree with columns ['in_degree_freq', 'out_degree_freq'].
    """
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    in_series = pd.Series(in_degrees).value_counts().rename("in_degree_freq")
    out_series = pd.Series(out_degrees).value_counts().rename("out_degree_freq")

    df_dist = pd.concat([in_series, out_series], axis=1).fillna(0).astype(int)
    df_dist.index.name = "degree"
    df_dist = df_dist.sort_index()
    return df_dist


def get_degree_percentiles(G: nx.DiGraph) -> pd.DataFrame:
    """Generate statistical percentiles for degree distributions.

    Demonstrates heavy-tailed / power-law scale-free properties in Wikipedia voting.
    """
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    percentiles = [10, 25, 50, 75, 90, 95, 99, 100]
    in_pct = np.percentile(in_degrees, percentiles)
    out_pct = np.percentile(out_degrees, percentiles)

    df_pct = pd.DataFrame(
        {
            "Percentile": [f"P{p}" for p in percentiles],
            "In-Degree (Votes Received)": in_pct,
            "Out-Degree (Votes Cast)": out_pct,
        }
    )
    return df_pct
