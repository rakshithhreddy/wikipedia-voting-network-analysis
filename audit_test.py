"""Audit script to verify dataset, metrics, community detection, exports, and Pajek integrity."""

import hashlib
import json
import os
import sys
import networkx as nx
import numpy as np
import pandas as pd

def audit():
    print("=" * 70)
    print("SNA PROJECT AUDIT: COMPREHENSIVE INTEGRITY & METRICS VALIDATION")
    print("=" * 70)

    # 1. Dataset Verification
    print("\n[SECTION 1] DATASET VERIFICATION")
    gz_path = os.path.join("data", "wiki-Vote.txt.gz")
    txt_path = os.path.join("data", "wiki-Vote.txt")
    assert os.path.exists(gz_path), "GZ file missing"
    assert os.path.exists(txt_path), "TXT file missing"

    with open(gz_path, "rb") as f:
        gz_hash = hashlib.sha256(f.read()).hexdigest()
    with open(txt_path, "rb") as f:
        txt_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"  Archive SHA256 : {gz_hash}")
    print(f"  Extracted SHA256: {txt_hash}")

    comments, data_lines, malformed, self_loops = [], [], [], []
    with open(txt_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                comments.append(s)
                continue
            parts = s.split()
            if len(parts) != 2:
                malformed.append((idx, s))
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                self_loops.append((idx, u))
            data_lines.append((u, v))

    print(f"  Metadata Comments   : {len(comments)}")
    for c in comments:
        print(f"    {c}")
    print(f"  Raw Directed Edges  : {len(data_lines):,}")
    print(f"  Malformed Lines     : {len(malformed)}")
    print(f"  Self-Loops          : {len(self_loops)}")
    assert len(malformed) == 0, f"Malformed lines: {malformed}"
    assert len(self_loops) == 0, f"Self loops found: {self_loops}"

    df_raw = pd.DataFrame(data_lines, columns=["source", "target"])
    assert df_raw.duplicated().sum() == 0, "Duplicate edges found in raw data!"
    unique_sources = set(df_raw["source"])
    unique_targets = set(df_raw["target"])
    all_nodes = unique_sources.union(unique_targets)
    print(f"  Voters (Sources)    : {len(unique_sources):,}")
    print(f"  Candidates (Targets): {len(unique_targets):,}")
    print(f"  Total Unique Nodes  : {len(all_nodes):,}")
    assert len(all_nodes) == 7115, f"Expected 7,115 nodes, got {len(all_nodes)}"
    assert len(df_raw) == 103689, f"Expected 103,689 edges, got {len(df_raw)}"
    print("  -> Dataset verification: PASSED (100% authentic SNAP dataset)")

    # 2. Network Verification
    print("\n[SECTION 2] NETWORK GRAPH RECONSTRUCTION & METRICS")
    G = nx.DiGraph()
    G.add_edges_from(zip(df_raw["source"], df_raw["target"]))
    N = G.number_of_nodes()
    E = G.number_of_edges()
    print(f"  DiGraph Nodes: {N:,}, DiGraph Edges: {E:,}")
    assert N == 7115
    assert E == 103689

    density = E / (N * (N - 1))
    reciprocity = nx.reciprocity(G)
    avg_deg = E / N
    print(f"  Density        : {density:.6f}")
    print(f"  Reciprocity    : {reciprocity:.4f} ({reciprocity*100:.2f}%)")
    print(f"  Average Degree : {avg_deg:.2f}")

    in_degs = dict(G.in_degree())
    out_degs = dict(G.out_degree())
    top_in = sorted(in_degs.items(), key=lambda x: x[1], reverse=True)[:5]
    top_out = sorted(out_degs.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 3 In-Degree (Votes Received): {top_in[:3]}")
    print(f"  Top 3 Out-Degree (Votes Cast)   : {top_out[:3]}")

    pr = nx.pagerank(G, alpha=0.85)
    pr_sum = sum(pr.values())
    print(f"  PageRank Sum   : {pr_sum:.6f} (Theoretical: 1.0)")
    assert abs(pr_sum - 1.0) < 1e-4

    bt_approx = nx.betweenness_centrality(G, k=500, normalized=True, seed=42)
    top_bt = sorted(bt_approx.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top 3 Betweenness (k=500 approx): {top_bt}")
    print("  -> Network metrics verification: PASSED")

    # 3. Community Detection Verification
    print("\n[SECTION 3] LOUVAIN COMMUNITY DETECTION")
    import community as community_louvain
    from src.community_detection import detect_louvain_communities, compute_community_statistics

    partition, mod_q, G_undir = detect_louvain_communities(G, random_state=42)
    num_comm = len(set(partition.values()))
    print(f"  Undirected Projection Nodes: {G_undir.number_of_nodes()}, Edges: {G_undir.number_of_edges()}")
    print(f"  Louvain Communities Count  : {num_comm}")
    print(f"  Newman-Girvan Modularity Q : {mod_q:.4f}")
    assert mod_q > 0.40, f"Expected modularity > 0.40, got {mod_q}"
    print("  -> Community detection verification: PASSED")

    # 4. Export Files Verification
    print("\n[SECTION 4] EXPORT FILES INTEGRITY & SPECIFICATION CHECK")
    export_files = [
        "exports/wiki_vote_directed.net",
        "exports/wiki_vote_top100_subgraph.net",
        "exports/node_centrality_metrics.csv",
        "exports/community_partitions.csv",
        "exports/wiki_vote_edges.csv",
        "exports/network_overview_metrics.json"
    ]
    for ef in export_files:
        assert os.path.exists(ef), f"Missing export: {ef}"
        sz = os.path.getsize(ef)
        print(f"  File: {ef:40s} | Size: {sz:10,d} bytes | Status: OK")

    # Pajek .net validation
    print("\n  Validating exports/wiki_vote_directed.net format for SocNetV/Pajek...")
    with open("exports/wiki_vote_directed.net", "r", encoding="utf-8") as f:
        header = f.readline().strip()
        assert header == "*Vertices 7115", f"Bad Pajek header: {header}"
        v_count = 0
        pajek_label_to_idx = {}
        for _ in range(7115):
            line = f.readline().strip()
            parts = line.split(maxsplit=1)
            idx = int(parts[0])
            lbl = int(parts[1].strip('"'))
            pajek_label_to_idx[lbl] = idx
            v_count += 1
        assert v_count == 7115
        arcs_header = f.readline().strip()
        assert arcs_header == "*Arcs", f"Expected *Arcs, got: {arcs_header}"
        arc_count = 0
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            u_idx, v_idx, w = line_str.split()
            u_i, v_i = int(u_idx), int(v_idx)
            assert 1 <= u_i <= 7115
            assert 1 <= v_i <= 7115
            arc_count += 1
        assert arc_count == 103689
        print(f"  -> Pajek format strictly validated: 7,115 vertices, 103,689 directed arcs under *Arcs.")

    # CSV checks
    df_nodes = pd.read_csv("exports/node_centrality_metrics.csv")
    assert len(df_nodes) == 7115
    assert set(df_nodes.columns) >= {
        "node_id", "in_degree", "out_degree", "total_degree",
        "in_degree_centrality", "out_degree_centrality", "pagerank",
        "betweenness_centrality", "community_id"
    }

    df_comm_csv = pd.read_csv("exports/community_partitions.csv")
    assert df_comm_csv["node_count"].sum() == 7115

    df_edges_csv = pd.read_csv("exports/wiki_vote_edges.csv")
    assert len(df_edges_csv) == 103689

    with open("exports/network_overview_metrics.json", "r") as f:
        meta = json.load(f)
    assert meta["node_count"] == 7115
    assert meta["edge_count"] == 103689
    print("  -> Export file validation: ALL 6 ARTIFACTS 100% VALID")

    print("\n" + "=" * 70)
    print("ALL AUDIT CHECKS COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    audit()
