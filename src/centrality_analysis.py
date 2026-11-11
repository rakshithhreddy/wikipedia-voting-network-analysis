"""Centrality Analysis Module for Wikipedia Voting Network.

Computes and analyzes node-level centrality metrics on the directed graph:
- In-Degree Centrality (Prestige / Popularity of Candidates)
- Out-Degree Centrality (Electoral Participation / Voting Activity)
- PageRank Centrality (Endorsement Quality / Recursive Influence)
- Betweenness Centrality (Social Brokerage / Bridge Nodes) with documented Brandes k-sampling approximation.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def compute_all_centralities(
    G: nx.DiGraph,
    betweenness_k: int = 500,
    random_seed: int = 42,
    cache_path: Optional[str] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Compute comprehensive centrality metrics for all nodes in the directed graph.

    Approximation Documentation:
    --------------------------
    - In-Degree Centrality: EXACT. Normalized by (N - 1).
    - Out-Degree Centrality: EXACT. Normalized by (N - 1).
    - PageRank: EXACT power iteration algorithm (alpha=0.85, max_iter=100, tol=1e-06).
    - Betweenness Centrality: APPROXIMATION using the Brandes & Pich (2007)
      k-sampling algorithm (networkx.betweenness_centrality with k pivot nodes).
      Exact betweenness requires O(V * E) shortest path computations across all
      7,115 vertices (~7.38 * 10^8 operations). Sampling k=500 pivots provides an
      unbiased Monte Carlo estimator scaled by (N / k) that preserves relative
      centrality rankings while completing in seconds.

    Parameters
    ----------
    G : nx.DiGraph
        Directed voting network.
    betweenness_k : int
        Number of pivot sample nodes for betweenness approximation.
        If None, full exact computation is performed.
    random_seed : int
        Seed for reproducibility of betweenness pivot sampling.
    cache_path : Optional[str]
        File path to load/save precomputed metrics CSV.
    force_recompute : bool
        If True, ignores existing cache and recomputes all metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['node_id', 'in_degree', 'out_degree', 'total_degree',
         'in_degree_centrality', 'out_degree_centrality',
         'pagerank', 'betweenness_centrality',
         'rank_in_degree', 'rank_pagerank', 'rank_betweenness']
    """
    if (
        cache_path
        and os.path.exists(cache_path)
        and os.path.getsize(cache_path) > 0
        and not force_recompute
    ):
        logger.info(f"Loading precomputed centrality metrics from {cache_path}...")
        df_cached = pd.read_csv(cache_path)
        return df_cached

    logger.info("Computing degree metrics...")
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    in_deg_centrality = nx.in_degree_centrality(G)
    out_deg_centrality = nx.out_degree_centrality(G)

    logger.info("Computing PageRank (alpha=0.85)...")
    pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)

    logger.info(
        f"Computing Betweenness Centrality (Brandes approximation with k={betweenness_k}, seed={random_seed})..."
    )
    # Ulrik Brandes k-sampling approximation
    bt_scores = nx.betweenness_centrality(
        G,
        k=betweenness_k,
        normalized=True,
        seed=random_seed,
    )

    records = []
    for node in G.nodes():
        in_d = in_degrees.get(node, 0)
        out_d = out_degrees.get(node, 0)
        records.append(
            {
                "node_id": int(node),
                "in_degree": in_d,
                "out_degree": out_d,
                "total_degree": in_d + out_d,
                "in_degree_centrality": in_deg_centrality.get(node, 0.0),
                "out_degree_centrality": out_deg_centrality.get(node, 0.0),
                "pagerank": pagerank_scores.get(node, 0.0),
                "betweenness_centrality": bt_scores.get(node, 0.0),
            }
        )

    df = pd.DataFrame(records)

    # Compute rankings (1 is highest)
    df["rank_in_degree"] = df["in_degree"].rank(ascending=False, method="min").astype(int)
    df["rank_out_degree"] = df["out_degree"].rank(ascending=False, method="min").astype(int)
    df["rank_pagerank"] = df["pagerank"].rank(ascending=False, method="min").astype(int)
    df["rank_betweenness"] = (
        df["betweenness_centrality"].rank(ascending=False, method="min").astype(int)
    )

    # Sort primarily by in_degree
    df = df.sort_values(by="in_degree", ascending=False).reset_index(drop=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"Saved computed centrality metrics to {cache_path}")

    return df


def get_top_k_users(
    df_metrics: pd.DataFrame, metric: str = "in_degree", k: int = 10
) -> pd.DataFrame:
    """Retrieve top K users ranked by the specified centrality metric.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        DataFrame of node centrality metrics.
    metric : str
        Metric column to sort by (e.g. 'in_degree', 'pagerank', 'betweenness_centrality').
    k : int
        Number of top users to return.

    Returns
    -------
    pd.DataFrame
        Top K ranked nodes.
    """
    valid_cols = [
        "in_degree",
        "out_degree",
        "total_degree",
        "in_degree_centrality",
        "out_degree_centrality",
        "pagerank",
        "betweenness_centrality",
    ]
    if metric not in valid_cols:
        raise ValueError(f"Unknown metric '{metric}'. Expected one of {valid_cols}")

    return df_metrics.sort_values(by=metric, ascending=False).head(k).reset_index(drop=True)


def compute_centrality_correlations(df_metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman rank correlation between centrality measures.

    In Social Network Analysis, comparing Pearson (linear) and Spearman (monotonic rank)
    correlations reveals whether popularity (in-degree) directly translates to structural
    prestige (PageRank) or bridging influence (betweenness).
    """
    cols = [
        "in_degree",
        "out_degree",
        "total_degree",
        "pagerank",
        "betweenness_centrality",
    ]
    sub_df = df_metrics[cols].rename(
        columns={
            "in_degree": "In-Degree",
            "out_degree": "Out-Degree",
            "total_degree": "Total Degree",
            "pagerank": "PageRank",
            "betweenness_centrality": "Betweenness",
        }
    )
    pearson_corr = sub_df.corr(method="pearson")
    spearman_corr = sub_df.corr(method="spearman")
    return pearson_corr, spearman_corr
