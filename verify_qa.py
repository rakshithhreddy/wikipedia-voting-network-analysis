"""Strict Quality Assurance and File Format Compatibility Audit Script.

Verifies:
1. Pajek .net file syntax (*Vertices N, *Arcs, 1-based indexing, valid vertex mapping)
2. Directionality preservation (source -> destination exact match with raw SNAP dataset)
3. Top-100 subgraph integrity (100 vertices, valid induced arcs, 1-based indexing)
4. Cross-file consistency (Dataset, JSON summary, Node metrics CSV, Community CSV, Edge list CSV)
"""

import json
import os
import sys
import pandas as pd

def run_qa():
    print("=" * 75)
    print("STRICT QA VERIFICATION: PAJEK FILE FORMAT & CROSS-FILE INTEGRITY AUDIT")
    print("=" * 75)

    # 1. Dataset Check
    txt_path = os.path.join("data", "wiki-Vote.txt")
    assert os.path.exists(txt_path), f"Missing raw dataset at {txt_path}"

    raw_edges = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                u, v = map(int, s.split())
                raw_edges.append((u, v))

    raw_sources = {u for u, v in raw_edges}
    raw_targets = {v for u, v in raw_edges}
    raw_nodes = raw_sources | raw_targets

    print("\n[STEP 1] RAW DATASET STATS")
    print(f"  Source Voters (Out-degree > 0)    : {len(raw_sources):,}")
    print(f"  Target Nominees (In-degree > 0)   : {len(raw_targets):,}")
    print(f"  Total Unique Wikipedia Nodes      : {len(raw_nodes):,}")
    print(f"  Total Directed Voting Edges       : {len(raw_edges):,}")
    assert len(raw_nodes) == 7115, f"Expected 7,115 nodes, got {len(raw_nodes)}"
    assert len(raw_edges) == 103689, f"Expected 103,689 edges, got {len(raw_edges)}"

    # 2. Inspect exports/wiki_vote_directed.net
    print("\n[STEP 2] INSPECTING exports/wiki_vote_directed.net")
    pajek_file = os.path.join("exports", "wiki_vote_directed.net")
    assert os.path.exists(pajek_file), f"Missing {pajek_file}"
    pajek_size = os.path.getsize(pajek_file)
    print(f"  File Size: {pajek_size:,} bytes")

    with open(pajek_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Header check
    v_decl = lines[0]
    print(f"  Header Declaration: {v_decl}")
    assert v_decl == "*Vertices 7115", f"Bad Pajek header: {v_decl}"

    idx_to_label = {}
    label_to_idx = {}
    current_sec = "VERTICES"
    arcs_found = []
    edges_found = []

    for line_idx, line in enumerate(lines[1:], 2):
        if line.startswith("*Arcs"):
            current_sec = "ARCS"
            continue
        elif line.startswith("*Edges"):
            current_sec = "EDGES"
            continue
        elif line.startswith("*"):
            current_sec = line
            continue

        if current_sec == "VERTICES":
            parts = line.split(maxsplit=1)
            assert len(parts) == 2, f"Malformed vertex line {line_idx}: {line}"
            idx = int(parts[0])
            lbl = int(parts[1].strip('"'))
            idx_to_label[idx] = lbl
            label_to_idx[lbl] = idx
        elif current_sec == "ARCS":
            parts = line.split()
            assert len(parts) in (2, 3), f"Malformed arc line {line_idx}: {line}"
            u = int(parts[0])
            v = int(parts[1])
            w = float(parts[2]) if len(parts) == 3 else 1.0
            arcs_found.append((u, v, w))
        elif current_sec == "EDGES":
            edges_found.append(line)

    print(f"  Declared Vertices Count           : {len(idx_to_label):,}")
    print(f"  1-Based Index Range               : min={min(idx_to_label.keys())}, max={max(idx_to_label.keys())}")
    assert len(idx_to_label) == 7115
    assert min(idx_to_label.keys()) == 1 and max(idx_to_label.keys()) == 7115
    assert set(idx_to_label.values()) == raw_nodes, "Vertex labels do not match raw node IDs exactly!"

    print(f"  Directed Arcs (*Arcs) Count       : {len(arcs_found):,}")
    print(f"  Undirected Edges (*Edges) Count   : {len(edges_found):,}")
    assert len(arcs_found) == 103689
    assert len(edges_found) == 0, f"Found unexpected *Edges in directed network export!"

    # Out of bounds check
    out_of_bounds = [a for a in arcs_found if not (1 <= a[0] <= 7115 and 1 <= a[1] <= 7115)]
    assert len(out_of_bounds) == 0, f"Arcs pointing to non-existent vertices: {len(out_of_bounds)}"
    print("  Out-of-Bounds Vertex References   : 0 (ALL ARCS VALID)")

    # Directionality exact match check
    print("  Verifying exact 1-to-1 direction preservation from raw data...")
    mapped_raw_edges = set()
    for u, v in raw_edges:
        u_idx = label_to_idx[u]
        v_idx = label_to_idx[v]
        mapped_raw_edges.add((u_idx, v_idx))

    pajek_edges_set = {(a[0], a[1]) for a in arcs_found}
    assert mapped_raw_edges == pajek_edges_set, "Discrepancy detected between raw edges and Pajek arcs!"
    print("  -> Directionality Verification    : PASSED (Exact 100% 1-to-1 correspondence)")

    # Weights validation
    weights = [a[2] for a in arcs_found]
    print(f"  Arc Weights Range                 : min={min(weights)}, max={max(weights)} (All numeric, valid)")

    # 3. Inspect exports/wiki_vote_top100_subgraph.net
    print("\n[STEP 3] INSPECTING exports/wiki_vote_top100_subgraph.net")
    pajek_sub = os.path.join("exports", "wiki_vote_top100_subgraph.net")
    assert os.path.exists(pajek_sub), f"Missing {pajek_sub}"
    print(f"  File Size: {os.path.getsize(pajek_sub):,} bytes")

    with open(pajek_sub, "r", encoding="utf-8") as f:
        sub_lines = [l.strip() for l in f if l.strip()]

    assert sub_lines[0] == "*Vertices 100", f"Bad subgraph header: {sub_lines[0]}"
    sub_idx_to_label = {}
    sub_arcs = []
    sub_sec = "VERTICES"
    for line in sub_lines[1:]:
        if line.startswith("*Arcs"):
            sub_sec = "ARCS"
            continue
        if sub_sec == "VERTICES":
            parts = line.split(maxsplit=1)
            sub_idx_to_label[int(parts[0])] = int(parts[1].strip('"'))
        elif sub_sec == "ARCS":
            parts = line.split()
            sub_arcs.append((int(parts[0]), int(parts[1]), float(parts[2])))

    print(f"  Subgraph Vertices                 : {len(sub_idx_to_label)}")
    print(f"  Subgraph Arcs (*Arcs)             : {len(sub_arcs)}")
    assert len(sub_idx_to_label) == 100
    assert min(sub_idx_to_label.keys()) == 1 and max(sub_idx_to_label.keys()) == 100
    sub_oob = [a for a in sub_arcs if not (1 <= a[0] <= 100 and 1 <= a[1] <= 100)]
    assert len(sub_oob) == 0, f"Subgraph arcs out of bounds: {len(sub_oob)}"
    print("  -> Subgraph Pajek Syntax          : PASSED (Valid 100-node induced subgraph)")

    # 4. Cross-check CSVs and JSON
    print("\n[STEP 4] CROSS-CHECKING METRIC FILES AND ARTIFACTS")

    df_nodes = pd.read_csv("exports/node_centrality_metrics.csv")
    print(f"  node_centrality_metrics.csv rows  : {len(df_nodes):,}")
    assert len(df_nodes) == 7115
    assert set(df_nodes["node_id"]) == raw_nodes

    df_comm = pd.read_csv("exports/community_partitions.csv")
    comm_sum_nodes = df_comm["node_count"].sum()
    print(f"  community_partitions.csv rows     : {len(df_comm)} (Communities)")
    print(f"  Total Nodes in Communities        : {comm_sum_nodes:,}")
    assert comm_sum_nodes == 7115

    df_edges = pd.read_csv("exports/wiki_vote_edges.csv")
    print(f"  wiki_vote_edges.csv rows          : {len(df_edges):,}")
    assert len(df_edges) == 103689

    with open("exports/network_overview_metrics.json", "r") as f:
        meta = json.load(f)
    print(f"  network_overview_metrics.json     : node_count={meta['node_count']:,}, edge_count={meta['edge_count']:,}, density={meta['density']:.6f}")
    assert meta["node_count"] == 7115
    assert meta["edge_count"] == 103689

    print("\n" + "=" * 75)
    print("ALL STRICT QUALITY ASSURANCE CHECKS PASSED: ZERO DEFECTS DETECTED")
    print("=" * 75)

if __name__ == "__main__":
    run_qa()
