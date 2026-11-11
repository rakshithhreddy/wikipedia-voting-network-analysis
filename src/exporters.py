"""Exporters Module for Wikipedia Voting Network.

Generates standard Pajek (.net) and CSV data exports strictly conforming
to Social Network Analysis software standards (SocNetV, Pajek, Gephi, UCINET).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def export_to_pajek(
    G: nx.DiGraph,
    output_path: str,
    include_weights: bool = True,
) -> str:
    """Export directed graph G to standard Pajek (.net) format.

    SocNetV & Pajek Compatibility Assurance:
    ----------------------------------------
    1. Pajek requires 1-based sequential integer vertex indices (1 to N).
       Original node identifiers (e.g. 30, 1412) are mapped to 1..N and
       preserved accurately as string labels in quotes: `1 "30"`.
    2. Directed edges MUST be declared under the `*Arcs` section header.
       (In contrast, `*Edges` is reserved for undirected ties in Pajek/SocNetV).
    3. Each arc line specifies `u_idx v_idx weight`. Default weight is 1.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph to export.
    output_path : str
        Target destination path for .net file.
    include_weights : bool
        Whether to append edge weight column (default True).

    Returns
    -------
    str
        Output file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    nodes = sorted(list(G.nodes()))
    n_nodes = len(nodes)

    # 1-based indexing map
    node_to_idx = {node: idx + 1 for idx, node in enumerate(nodes)}

    logger.info(f"Writing {n_nodes} vertices and {G.number_of_edges()} arcs to Pajek: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        # Write Vertices block
        f.write(f"*Vertices {n_nodes}\n")
        for node in nodes:
            idx = node_to_idx[node]
            f.write(f'{idx} "{node}"\n')

        # Write Arcs block (Directed edges)
        f.write("*Arcs\n")
        for u, v, data in G.edges(data=True):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            w = data.get("weight", 1)
            if include_weights:
                f.write(f"{u_idx} {v_idx} {w}\n")
            else:
                f.write(f"{u_idx} {v_idx}\n")

    logger.info(f"Pajek .net successfully exported: {os.path.getsize(output_path):,} bytes.")
    return output_path


def export_node_metrics_csv(df_metrics: pd.DataFrame, output_path: str) -> str:
    """Export complete node-level centrality & community metrics to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_metrics.to_csv(output_path, index=False)
    logger.info(f"Exported node metrics CSV to {output_path} ({len(df_metrics)} rows)")
    return output_path


def export_community_summary_csv(df_community: pd.DataFrame, output_path: str) -> str:
    """Export community detection breakdown to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_community.to_csv(output_path, index=False)
    logger.info(f"Exported community summary CSV to {output_path} ({len(df_community)} rows)")
    return output_path


def export_edge_list_csv(G: nx.DiGraph, output_path: str) -> str:
    """Export directed edge list to CSV format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    edges = list(G.edges(data=True))
    records = [
        {
            "source_voter": u,
            "target_candidate": v,
            "weight": data.get("weight", 1),
        }
        for u, v, data in edges
    ]
    df_edges = pd.DataFrame(records)
    df_edges.to_csv(output_path, index=False)
    logger.info(f"Exported edge list CSV to {output_path} ({len(df_edges)} rows)")
    return output_path


def export_global_metrics_json(metrics: Dict[str, Any], output_path: str) -> str:
    """Export global network summary metrics to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Exported global metrics JSON to {output_path}")
    return output_path
