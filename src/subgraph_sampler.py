"""Subgraph Sampler Module for Wikipedia Voting Network.

Extracts meaningful, manageable subgraphs from the full 7,115-node / 103,689-edge
network for responsive, interactive visualization in Plotly / Streamlit.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def extract_top_influencers_subgraph(
    G: nx.DiGraph,
    df_metrics: pd.DataFrame,
    top_k: int = 60,
    metric: str = "in_degree",
) -> nx.DiGraph:
    """Extract the induced subgraph among the top K most influential users.

    Reveals the voting core / elite interactions within the Wikipedia hierarchy.

    Parameters
    ----------
    G : nx.DiGraph
        Original directed network.
    df_metrics : pd.DataFrame
        Centrality metrics table.
    top_k : int
        Number of top nodes to include.
    metric : str
        Metric column to sort by ('in_degree', 'pagerank', 'betweenness_centrality').

    Returns
    -------
    nx.DiGraph
        Induced directed subgraph.
    """
    top_nodes = list(
        df_metrics.sort_values(by=metric, ascending=False).head(top_k)["node_id"]
    )
    subgraph = G.subgraph(top_nodes).copy()
    logger.info(
        f"Extracted Top-{top_k} ({metric}) subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges."
    )
    return subgraph


def extract_ego_subgraph(
    G: nx.DiGraph,
    center_node: int,
    radius: int = 1,
    max_neighbors: int = 80,
) -> nx.DiGraph:
    """Extract the ego network surrounding a specific user.

    Includes the center node, their voters (predecessors), nominees they voted for
    (successors), and mutual edges among them.

    Parameters
    ----------
    G : nx.DiGraph
        Full directed graph.
    center_node : int
        Target user node ID.
    radius : int
        Hop radius (default 1).
    max_neighbors : int
        Cap on neighbors to preserve visual readability.

    Returns
    -------
    nx.DiGraph
        Ego directed subgraph.
    """
    if center_node not in G:
        raise ValueError(f"Node {center_node} does not exist in graph.")

    # Get voters (predecessors) and voted-for (successors)
    voters = list(G.predecessors(center_node))
    voted_for = list(G.successors(center_node))

    # If too many neighbors, prioritize top in-degree neighbors
    combined_neighbors = list(set(voters + voted_for))
    if len(combined_neighbors) > max_neighbors:
        combined_neighbors.sort(key=lambda n: G.degree(n), reverse=True)
        selected_nodes = set(combined_neighbors[:max_neighbors])
    else:
        selected_nodes = set(combined_neighbors)

    selected_nodes.add(center_node)
    ego_graph = G.subgraph(selected_nodes).copy()
    logger.info(
        f"Extracted Ego subgraph for node {center_node}: {ego_graph.number_of_nodes()} nodes, {ego_graph.number_of_edges()} edges."
    )
    return ego_graph


def extract_community_subgraph(
    G: nx.DiGraph,
    partition: Dict[int, int],
    community_id: int,
    max_nodes: int = 80,
) -> nx.DiGraph:
    """Extract an induced subgraph for a specific detected Louvain community.

    Parameters
    ----------
    G : nx.DiGraph
        Full directed graph.
    partition : Dict[int, int]
        Node to community mapping.
    community_id : int
        Community ID to isolate.
    max_nodes : int
        Maximum nodes to include (sampled by degree if larger).

    Returns
    -------
    nx.DiGraph
        Community induced subgraph.
    """
    comm_nodes = [node for node, c in partition.items() if c == community_id]
    if not comm_nodes:
        raise ValueError(f"Community {community_id} contains no nodes.")

    if len(comm_nodes) > max_nodes:
        comm_nodes.sort(key=lambda n: G.degree(n), reverse=True)
        selected_nodes = comm_nodes[:max_nodes]
    else:
        selected_nodes = comm_nodes

    comm_graph = G.subgraph(selected_nodes).copy()
    logger.info(
        f"Extracted Community {community_id} subgraph: {comm_graph.number_of_nodes()} nodes, {comm_graph.number_of_edges()} edges."
    )
    return comm_graph


def extract_degree_filtered_subgraph(
    G: nx.DiGraph,
    min_degree: int = 50,
    max_nodes: int = 150,
) -> nx.DiGraph:
    """Extract subgraph of high-activity nodes exceeding a degree threshold."""
    qualifying_nodes = [node for node, deg in G.degree() if deg >= min_degree]
    if len(qualifying_nodes) > max_nodes:
        qualifying_nodes.sort(key=lambda n: G.degree(n), reverse=True)
        qualifying_nodes = qualifying_nodes[:max_nodes]

    sub = G.subgraph(qualifying_nodes).copy()
    logger.info(
        f"Extracted Degree >= {min_degree} subgraph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges."
    )
    return sub
