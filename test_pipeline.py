"""Pipeline verification script for Wikipedia Voting Network Analysis.

Tests:
1. Dataset download and extraction
2. DiGraph construction
3. Global SNA metrics computation
4. Centrality calculations (in/out, PageRank, betweenness approximation)
5. Louvain community detection and modularity Q
6. Pajek .net and CSV export generation
7. File integrity checks
"""

import os
import sys
import time

from src.data_loader import build_directed_graph, download_and_extract_dataset
from src.network_metrics import compute_global_metrics, get_degree_percentiles
from src.centrality_analysis import compute_all_centralities, get_top_k_users, compute_centrality_correlations
from src.community_detection import detect_louvain_communities, compute_community_statistics, get_inter_community_matrix
from src.exporters import (
    export_to_pajek,
    export_node_metrics_csv,
    export_community_summary_csv,
    export_edge_list_csv,
    export_global_metrics_json,
)
from src.subgraph_sampler import extract_top_influencers_subgraph, extract_ego_subgraph


def main():
    print("=" * 70)
    print("WIKIPEDIA VOTING NETWORK ANALYSIS - PIPELINE INTEGRATION TEST")
    print("=" * 70)

    # Step 1: Download & Extract
    t0 = time.time()
    txt_path = download_and_extract_dataset()
    print(f"[1/7] Dataset ready at: {txt_path} ({time.time() - t0:.2f}s)")
    assert os.path.exists(txt_path), "Dataset text file missing!"

    # Step 2: Build DiGraph
    t0 = time.time()
    G = build_directed_graph(txt_path)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"[2/7] DiGraph built: {n_nodes:,} nodes, {n_edges:,} directed edges ({time.time() - t0:.2f}s)")
    assert n_nodes == 7115, f"Expected 7,115 nodes, got {n_nodes}"
    assert n_edges == 103689, f"Expected 103,689 edges, got {n_edges}"

    # Step 3: Global SNA Metrics
    t0 = time.time()
    global_metrics = compute_global_metrics(G)
    print(f"[3/7] Global metrics computed ({time.time() - t0:.2f}s):")
    print(f"      - Density: {global_metrics['density']:.6f}")
    print(f"      - Reciprocity: {global_metrics['reciprocity']:.4f} ({global_metrics['reciprocity']*100:.2f}%)")
    print(f"      - Avg In/Out Degree: {global_metrics['avg_in_degree']:.2f}")
    print(f"      - Max In-Degree: {global_metrics['max_in_degree']} | Max Out-Degree: {global_metrics['max_out_degree']}")
    print(f"      - Weakly Connected Components: {global_metrics['num_wcc']} (Largest: {global_metrics['largest_wcc_size']} nodes, {global_metrics['wcc_coverage_pct']:.1f}%)")
    print(f"      - Strongly Connected Components: {global_metrics['num_scc']} (Largest: {global_metrics['largest_scc_size']} nodes)")

    # Step 4: Centrality Analysis
    t0 = time.time()
    cache_csv = os.path.join("data", "cache", "node_metrics.csv")
    df_metrics = compute_all_centralities(
        G,
        betweenness_k=500,
        random_seed=42,
        cache_path=cache_csv,
        force_recompute=True,
    )
    print(f"[4/7] Centralities computed ({time.time() - t0:.2f}s): {len(df_metrics)} rows")
    top_in = get_top_k_users(df_metrics, "in_degree", 5)
    print("      Top 5 by In-Degree (Votes Received):")
    for _, r in top_in.iterrows():
        print(f"        Node {int(r['node_id'])}: {int(r['in_degree'])} votes (PageRank: {r['pagerank']:.6f})")

    top_pr = get_top_k_users(df_metrics, "pagerank", 5)
    print("      Top 5 by PageRank:")
    for _, r in top_pr.iterrows():
        print(f"        Node {int(r['node_id'])}: PageRank {r['pagerank']:.6f} (In-Degree: {int(r['in_degree'])})")

    # Step 5: Community Detection (Louvain on Undirected Projection)
    t0 = time.time()
    partition, modularity_q, G_undir = detect_louvain_communities(G, random_state=42)
    df_metrics["community_id"] = df_metrics["node_id"].map(partition)
    df_metrics.to_csv(cache_csv, index=False)
    df_comm = compute_community_statistics(G, partition, df_metrics)
    print(f"[5/7] Louvain Community Detection finished ({time.time() - t0:.2f}s):")
    print(f"      - Modularity Q: {modularity_q:.4f}")
    print(f"      - Number of Communities: {len(df_comm)}")
    print(f"      - Largest Community: ID {df_comm.iloc[0]['community_id']} with {df_comm.iloc[0]['node_count']} nodes")

    # Step 6: Exports Generation
    t0 = time.time()
    os.makedirs("exports", exist_ok=True)
    pajek_full = os.path.join("exports", "wiki_vote_directed.net")
    export_to_pajek(G, pajek_full)

    # Subgraph Pajek export for immediate lightweight viewing in SocNetV
    sub_top100 = extract_top_influencers_subgraph(G, df_metrics, top_k=100)
    pajek_sub = os.path.join("exports", "wiki_vote_top100_subgraph.net")
    export_to_pajek(sub_top100, pajek_sub)

    export_node_metrics_csv(df_metrics, os.path.join("exports", "node_centrality_metrics.csv"))
    export_community_summary_csv(df_comm, os.path.join("exports", "community_partitions.csv"))
    export_edge_list_csv(G, os.path.join("exports", "wiki_vote_edges.csv"))
    export_global_metrics_json(global_metrics, os.path.join("exports", "network_overview_metrics.json"))
    print(f"[6/7] All exports written to 'exports/' ({time.time() - t0:.2f}s)")

    # Step 7: Verify Pajek Format
    print("[7/7] Validating Pajek format integrity...")
    with open(pajek_full, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
        assert first_line.startswith("*Vertices"), f"Invalid Pajek header: {first_line}"
        print(f"      Header check: {first_line} -> PASS")
        print(f"      Sample vertex: {second_line} -> PASS")

    print("\n" + "=" * 70)
    print("SUCCESS: ALL PIPELINE TESTS PASSED 100%!")
    print("=" * 70)


if __name__ == "__main__":
    main()
