"""Community Detection Module for Wikipedia Voting Network.

Applies the Louvain heuristic algorithm on a documented undirected projection
of the directed voting network, evaluates Newman-Girvan modularity Q,
and calculates internal/external edge density and community leadership.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import community as community_louvain
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def detect_louvain_communities(
    G: nx.DiGraph,
    resolution: float = 1.0,
    random_state: int = 42,
    weight_bidirectional: bool = True,
) -> Tuple[Dict[int, int], float, nx.Graph]:
    """Execute Louvain community detection on an undirected projection of the directed graph.

    Methodology & Transformation Documentation:
    -------------------------------------------
    1. Theoretical Foundation:
       The standard Louvain algorithm (Blondel et al., 2008) optimizes the Newman-Girvan
       modularity index Q:
           Q = (1 / 2m) * sum_{i,j} [ A_{ij} - (k_i * k_j) / (2m) ] * delta(c_i, c_j)
       This objective function is natively defined for undirected networks where edges
       represent symmetric adjacency.

    2. Undirected Projection Strategy:
       In Wikipedia adminship voting, an edge (u -> v) is directional (voter -> candidate).
       To uncover voting coalitions and social affinity groups, the graph is projected
       into an undirected representation G_undir:
       - If user A voted for user B OR user B voted for user A, an undirected tie is formed.
       - If weight_bidirectional=True, reciprocal voting ties (A -> B and B -> A) are
         assigned a weight of 2, while unidirectional ties receive a weight of 1,
         thereby strengthening the affinity between mutually supportive editors.

    Parameters
    ----------
    G : nx.DiGraph
        Original directed voting network.
    resolution : float
        Resolution parameter for Louvain modularity. Defaults to 1.0.
    random_state : int
        Seed for reproducibility of node movement passes.
    weight_bidirectional : bool
        If True, reciprocal ties receive weight 2; otherwise weight 1.

    Returns
    -------
    Tuple[Dict[int, int], float, nx.Graph]
        - partition: Dict mapping node_id -> community_id
        - modularity_score: Newman-Girvan modularity Q
        - G_undir: Undirected projection graph with edge weights
    """
    logger.info("Constructing undirected weighted projection for Louvain community detection...")
    G_undir = nx.Graph()

    for u, v in G.edges():
        if G_undir.has_edge(u, v):
            # Reciprocal vote detected: increase edge weight
            if weight_bidirectional:
                G_undir[u][v]["weight"] = 2
        else:
            weight = 2 if (weight_bidirectional and G.has_edge(v, u)) else 1
            G_undir.add_edge(u, v, weight=weight)

    # Ensure all isolated nodes are present
    for n in G.nodes():
        if not G_undir.has_node(n):
            G_undir.add_node(n)

    logger.info(
        f"Undirected projection: {G_undir.number_of_nodes()} nodes, {G_undir.number_of_edges()} edges."
    )

    logger.info("Running Louvain optimization algorithm...")
    partition = community_louvain.best_partition(
        G_undir,
        weight="weight",
        resolution=resolution,
        random_state=random_state,
    )

    modularity_score = community_louvain.modularity(partition, G_undir, weight="weight")
    num_communities = len(set(partition.values()))
    logger.info(
        f"Louvain finished: found {num_communities} communities. Modularity Q = {modularity_score:.4f}"
    )

    return partition, modularity_score, G_undir


def compute_community_statistics(
    G: nx.DiGraph,
    partition: Dict[int, int],
    df_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Analyze properties of detected communities: size, internal edges, external edges,
    internal density, and top influential leader per community.

    Parameters
    ----------
    G : nx.DiGraph
        Directed voting graph.
    partition : Dict[int, int]
        Node to community mapping.
    df_metrics : Optional[pd.DataFrame]
        Node metrics containing PageRank/In-Degree to identify community leaders.

    Returns
    -------
    pd.DataFrame
        Summary table for each community sorted by node size descending.
    """
    # Group nodes by community
    community_nodes: Dict[int, List[int]] = {}
    for node, comm_id in partition.items():
        community_nodes.setdefault(comm_id, []).append(node)

    # Pre-index metrics for fast leader lookup
    node_to_pr = {}
    node_to_indeg = {}
    if df_metrics is not None:
        node_to_pr = dict(zip(df_metrics["node_id"], df_metrics["pagerank"]))
        node_to_indeg = dict(zip(df_metrics["node_id"], df_metrics["in_degree"]))

    records = []
    for comm_id, nodes in community_nodes.items():
        size = len(nodes)
        node_set = set(nodes)

        # Count internal and external directed edges
        internal_edges = 0
        outgoing_external = 0
        for u in nodes:
            for v in G.successors(u):
                if v in node_set:
                    internal_edges += 1
                else:
                    outgoing_external += 1

        # Directed internal density: internal_edges / (size * (size - 1))
        possible_internal = size * (size - 1)
        internal_density = (
            (internal_edges / possible_internal) if possible_internal > 0 else 0.0
        )

        # Find key leader in this community (highest PageRank or in_degree)
        if node_to_pr:
            leader_node = max(nodes, key=lambda n: node_to_pr.get(n, 0.0))
            leader_score = node_to_pr.get(leader_node, 0.0)
            leader_indeg = node_to_indeg.get(leader_node, 0)
        else:
            leader_node = max(nodes, key=lambda n: G.in_degree(n))
            leader_score = G.in_degree(leader_node)
            leader_indeg = leader_score

        records.append(
            {
                "community_id": comm_id,
                "node_count": size,
                "internal_edges": internal_edges,
                "outgoing_external_edges": outgoing_external,
                "internal_density": round(internal_density, 6),
                "leader_node_id": leader_node,
                "leader_pagerank": leader_score,
                "leader_in_degree": leader_indeg,
            }
        )

    df_comm = pd.DataFrame(records).sort_values(by="node_count", ascending=False).reset_index(drop=True)
    return df_comm


def get_inter_community_matrix(
    G: nx.DiGraph,
    partition: Dict[int, int],
    top_n_communities: int = 10,
) -> pd.DataFrame:
    """Construct an inter-community directed voting flow matrix for the top N communities.

    Matrix cell [i, j] represents the number of votes cast by members of
    Community i towards candidates in Community j.
    """
    # Determine top N communities by size
    sizes = pd.Series(partition).value_counts()
    top_comm_ids = list(sizes.head(top_n_communities).index)

    # Initialize zero matrix
    matrix = pd.DataFrame(
        0,
        index=[f"Comm {c}" for c in top_comm_ids],
        columns=[f"Comm {c}" for c in top_comm_ids],
    )

    comm_set = set(top_comm_ids)
    for u, v in G.edges():
        c_u = partition.get(u)
        c_v = partition.get(v)
        if c_u in comm_set and c_v in comm_set:
            matrix.loc[f"Comm {c_u}", f"Comm {c_v}"] += 1

    return matrix
