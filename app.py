"""Streamlit Dashboard for Wikipedia Voting Network Analysis.

Comprehensive Social Network Analytics application for analyzing Wikipedia
Request for Adminship (RfA) voting network from Stanford SNAP.
"""

import json
import os
import sys
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add current folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.centrality_analysis import (
    compute_all_centralities,
    compute_centrality_correlations,
    get_top_k_users,
)
from src.community_detection import (
    compute_community_statistics,
    detect_louvain_communities,
    get_inter_community_matrix,
)
from src.data_loader import build_directed_graph, download_and_extract_dataset
from src.exporters import (
    export_community_summary_csv,
    export_edge_list_csv,
    export_global_metrics_json,
    export_node_metrics_csv,
    export_to_pajek,
)
from src.network_metrics import (
    compute_global_metrics,
    get_degree_distribution_data,
    get_degree_percentiles,
)
from src.subgraph_sampler import (
    extract_community_subgraph,
    extract_degree_filtered_subgraph,
    extract_ego_subgraph,
    extract_top_influencers_subgraph,
)
from src.visualizer import create_plotly_network_figure

# Streamlit Page Configuration
st.set_page_config(
    page_title="Wikipedia Voting Network Analysis | SNA Lab",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Stanford SNAP Wiki-Vote Graph...")
def load_graph() -> nx.DiGraph:
    """Download and build directed graph."""
    txt_path = download_and_extract_dataset()
    return build_directed_graph(txt_path)


@st.cache_data(show_spinner="Loading SNA metrics & community partitions...")
def load_analysis_data(
    _G: nx.DiGraph,
) -> Tuple[Dict, pd.DataFrame, Dict[int, int], float, pd.DataFrame]:
    """Compute and cache all network metrics, centralities, and Louvain partitions."""
    exports_dir = "exports"
    cache_csv = os.path.join("data", "cache", "node_metrics.csv")
    node_metrics_path = os.path.join(exports_dir, "node_centrality_metrics.csv")
    comm_path = os.path.join(exports_dir, "community_partitions.csv")
    overview_json = os.path.join(exports_dir, "network_overview_metrics.json")

    # Fast path: Load pre-computed canonical data if available (ensures instant startup on Cloud)
    if (
        os.path.exists(node_metrics_path)
        and os.path.exists(comm_path)
        and os.path.exists(overview_json)
    ):
        with open(overview_json, "r", encoding="utf-8") as f:
            global_metrics = json.load(f)
        df_metrics = pd.read_csv(node_metrics_path)
        df_community = pd.read_csv(comm_path)
        partition = dict(zip(df_metrics["node_id"].astype(int), df_metrics["community_id"].astype(int)))
        
        import community as community_louvain
        G_undirected = _G.to_undirected()
        modularity_q = float(community_louvain.modularity(partition, G_undirected))
        return global_metrics, df_metrics, partition, modularity_q, df_community

    # Full computation fallback
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    global_metrics = compute_global_metrics(_G)

    # Centralities
    df_metrics = compute_all_centralities(
        _G,
        betweenness_k=500,
        random_seed=42,
        cache_path=cache_csv,
        force_recompute=not os.path.exists(cache_csv),
    )

    # Louvain communities on undirected projection
    partition, modularity_q, _ = detect_louvain_communities(_G, random_state=42)
    df_metrics["community_id"] = df_metrics["node_id"].map(partition)
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    df_metrics.to_csv(cache_csv, index=False)

    # Community statistics
    df_community = compute_community_statistics(_G, partition, df_metrics)

    # Export canonical files if missing
    os.makedirs("exports", exist_ok=True)
    pajek_path = os.path.join("exports", "wiki_vote_directed.net")
    if not os.path.exists(pajek_path):
        export_to_pajek(_G, pajek_path)
        export_node_metrics_csv(df_metrics, node_metrics_path)
        export_community_summary_csv(df_community, comm_path)
        export_edge_list_csv(_G, os.path.join("exports", "wiki_vote_edges.csv"))
        export_global_metrics_json(global_metrics, overview_json)

    return global_metrics, df_metrics, partition, modularity_q, df_community


# Initialize data
try:
    G = load_graph()
    global_metrics, df_metrics, partition, modularity_q, df_community = load_analysis_data(G)
except Exception as e:
    st.error(f"Error loading Wikipedia network data: {e}")
    st.stop()


# Sidebar
with st.sidebar:
    st.title("Wikipedia Voting SNA")
    st.markdown("**Stanford SNAP Wiki-Vote Dataset**")
    st.caption("Electoral relationships & social dynamics in Wikipedia administrator promotions.")

    st.divider()
    st.markdown("### Quick Statistics")
    st.markdown(f"- **Nodes (Users):** `{G.number_of_nodes():,}`")
    st.markdown(f"- **Edges (Votes):** `{G.number_of_edges():,}`")
    st.markdown(f"- **Density:** `{global_metrics['density']:.6f}`")
    st.markdown(f"- **Reciprocity:** `{global_metrics['reciprocity'] * 100:.2f}%`")
    st.markdown(f"- **Modularity (Q):** `{modularity_q:.4f}`")
    st.markdown(f"- **Communities:** `{len(df_community)}`")

    st.divider()
    st.markdown("### About the Research")
    st.info(
        "**Dataset Source:** [Stanford SNAP](https://snap.stanford.edu/data/wiki-Vote.html)\n\n"
        "Nodes represent Wikipedia users. A directed edge **(u → v)** indicates that user **u** "
        "cast a vote on candidate **v** for administrator promotion.\n\n"
        "Admin elections reflect social consensus, prestige, and community trust."
    )

    st.markdown("### Tech Stack")
    st.caption("Python 3.12 • NetworkX • Pandas • Plotly • Streamlit • Louvain")


# Main Dashboard Header
st.markdown(
    '<div class="main-header">Wikipedia Voting Network Analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">'
    "Social Network Analytics (SNA) of Wikipedia Request for Adminship (RfA) Electoral Structure"
    "</div>",
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🌐 Network Overview",
        "🏆 Influential Users",
        "🧩 Community Detection",
        "🕸️ Network Visualisation",
        "💾 Data & Exports",
    ]
)


# ==============================================================================
# TAB 1: NETWORK OVERVIEW
# ==============================================================================
with tab1:
    st.subheader("Global Network Topology & Macro Properties")
    st.markdown(
        "Summary of macroscopic structural metrics characterizing the Wikipedia RfA election network."
    )

    # Row 1: KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Users (Nodes)",
            value=f"{global_metrics['node_count']:,}",
            help="Total number of unique Wikipedia users who either cast or received a vote.",
        )
    with col2:
        st.metric(
            label="Directed Votes (Edges)",
            value=f"{global_metrics['edge_count']:,}",
            help="Total count of directed voting interactions recorded in RfA elections.",
        )
    with col3:
        st.metric(
            label="Network Density",
            value=f"{global_metrics['density']:.6f}",
            help="Fraction of all possible directed edges that actually exist: E / (N * (N-1)).",
        )
    with col4:
        st.metric(
            label="Reciprocity Rate",
            value=f"{global_metrics['reciprocity'] * 100:.2f}%",
            help="Percentage of mutual voting pairs: cases where user A voted for user B and B voted for A.",
        )

    # Row 2: Secondary KPIs
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(
            label="Avg Degree (In/Out)",
            value=f"{global_metrics['avg_in_degree']:.2f}",
            help="Mean number of votes cast per user = mean votes received per user.",
        )
    with col6:
        st.metric(
            label="Weakly Connected Coverage",
            value=f"{global_metrics['wcc_coverage_pct']:.1f}%",
            help=f"Largest WCC contains {global_metrics['largest_wcc_size']:,} of {global_metrics['node_count']:,} users.",
        )
    with col7:
        st.metric(
            label="Strongly Connected Core",
            value=f"{global_metrics['largest_scc_size']:,} users",
            help=f"Size of largest SCC ({global_metrics['scc_coverage_pct']:.1f}%): mutual reachable voting cycles.",
        )
    with col8:
        st.metric(
            label="Louvain Modularity (Q)",
            value=f"{modularity_q:.4f}",
            help="Newman-Girvan modularity score on the undirected projection.",
        )

    st.divider()

    # Degree Distribution Analysis
    st.markdown("### Degree Distributions & Scale-Free Network Properties")
    st.markdown(
        """
        In Social Network Analysis, degree distribution reveals whether a social system follows
        a Poisson / random graph model or a **scale-free power-law** regime.
        In Wikipedia voting:
        - **In-Degree (Votes Received):** Measures candidate prestige and endorsement volume.
        - **Out-Degree (Votes Cast):** Measures voter participation and civic activity.
        """
    )

    df_dist = get_degree_distribution_data(G)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # In-degree log-log plot
        in_subset = df_dist[df_dist["in_degree_freq"] > 0].reset_index()
        fig_in = px.scatter(
            in_subset,
            x="degree",
            y="in_degree_freq",
            log_x=True,
            log_y=True,
            labels={
                "degree": "In-Degree k_in (Votes Received)",
                "in_degree_freq": "Count of Users",
            },
            title="In-Degree Distribution (Log-Log Scale)",
            color_discrete_sequence=["#2563eb"],
        )
        fig_in.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_in, use_container_width=True)

    with col_chart2:
        # Out-degree log-log plot
        out_subset = df_dist[df_dist["out_degree_freq"] > 0].reset_index()
        fig_out = px.scatter(
            out_subset,
            x="degree",
            y="out_degree_freq",
            log_x=True,
            log_y=True,
            labels={"degree": "Out-Degree k_out (Votes Cast)", "out_degree_freq": "Count of Users"},
            title="Out-Degree Distribution (Log-Log Scale)",
            color_discrete_sequence=["#16a34a"],
        )
        fig_out.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_out, use_container_width=True)

    # Degree Percentiles & Role Breakdown
    col_pct, col_roles = st.columns([1.2, 1])

    with col_pct:
        st.markdown("#### Degree Distribution Percentiles")
        df_pct = get_degree_percentiles(G)
        st.dataframe(df_pct, use_container_width=True, hide_index=True)

    with col_roles:
        st.markdown("#### User Role Partitioning")
        role_df = pd.DataFrame(
            {
                "Role Category": [
                    "Active Dual Users (Voted & Received)",
                    "Pure Voters (Voted only, in-degree = 0)",
                    "Pure Candidates (Received only, out-degree = 0)",
                ],
                "Count": [
                    global_metrics["node_count"]
                    - global_metrics["pure_voters_count"]
                    - global_metrics["pure_candidates_count"],
                    global_metrics["pure_voters_count"],
                    global_metrics["pure_candidates_count"],
                ],
            }
        )
        fig_pie = px.pie(
            role_df,
            names="Role Category",
            values="Count",
            color_discrete_sequence=["#3b82f6", "#f59e0b", "#ec4899"],
            hole=0.45,
            title="User Electoral Participation Profile",
        )
        fig_pie.update_layout(height=280, margin=dict(t=35, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)


# ==============================================================================
# TAB 2: INFLUENTIAL USERS & CENTRALITY ANALYSIS
# ==============================================================================
with tab2:
    st.subheader("Influential Users & Multi-Metric Centrality Analysis")
    st.markdown(
        """
        Identify high-status administrators, active electorate members, and structural brokers
        using directed Social Network Analysis centrality metrics.
        """
    )

    # Filter Controls
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1.5, 1, 1.5])
    with c_ctrl1:
        rank_metric = st.selectbox(
            "Ranking Metric",
            options=[
                ("in_degree", "In-Degree (Candidate Popularity / Votes Received)"),
                ("pagerank", "PageRank (Prestige / High-Status Endorsements)"),
                ("betweenness_centrality", "Betweenness Centrality (Brandes Approx, k=500)"),
                ("out_degree", "Out-Degree (Electoral Activity / Votes Cast)"),
                ("total_degree", "Total Degree (Overall Connectivity)"),
            ],
            format_func=lambda x: x[1],
        )[0]

    with c_ctrl2:
        top_n = st.slider("Number of Top Users", min_value=5, max_value=50, value=10, step=5)

    with c_ctrl3:
        st.caption("Centrality Approximation Note:")
        st.caption(
            "Betweenness centrality uses Brandes & Pich (2007) random pivot sampling "
            "(k=500, seed=42) normalized by (N-1)(N-2). PageRank uses standard damping alpha=0.85."
        )

    # Top Users Table
    df_top = get_top_k_users(df_metrics, metric=rank_metric, k=top_n)

    col_tbl, col_bar = st.columns([1.2, 1])

    with col_tbl:
        st.markdown(f"#### Top {top_n} Users Ranked by `{rank_metric}`")
        display_cols = [
            "node_id",
            "in_degree",
            "out_degree",
            "pagerank",
            "betweenness_centrality",
            "community_id",
        ]
        format_dict = {
            "in_degree": "{:,}",
            "out_degree": "{:,}",
            "pagerank": "{:.6f}",
            "betweenness_centrality": "{:.6f}",
        }
        st.dataframe(
            df_top[display_cols].style.format(format_dict).background_gradient(
                subset=[rank_metric], cmap="Blues"
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col_bar:
        st.markdown(f"#### Centrality Comparison for Top {top_n}")
        fig_bar = px.bar(
            df_top,
            x="node_id",
            y=rank_metric,
            text_auto=True if rank_metric in ["in_degree", "out_degree", "total_degree"] else False,
            labels={"node_id": "User Node ID", rank_metric: rank_metric.replace("_", " ").title()},
            title=f"Top {top_n} Users by {rank_metric}",
            color=rank_metric,
            color_continuous_scale="Viridis",
        )
        fig_bar.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # Centrality Correlations & Cross-Metric Dynamics
    st.markdown("### Cross-Centrality Correlation & Trade-Off Dynamics")
    col_corr1, col_corr2 = st.columns(2)

    with col_corr1:
        # Scatter: In-Degree vs PageRank
        fig_scat1 = px.scatter(
            df_metrics.head(300),
            x="in_degree",
            y="pagerank",
            color="community_id",
            hover_name="node_id",
            hover_data=["in_degree", "out_degree", "pagerank", "betweenness_centrality"],
            labels={"in_degree": "In-Degree (Votes Received)", "pagerank": "PageRank"},
            title="In-Degree vs PageRank (Endorsement Quality vs Volume)",
            color_continuous_scale="Turbo",
        )
        fig_scat1.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_scat1, use_container_width=True)

    with col_corr2:
        # Scatter: In-Degree vs Out-Degree (Candidate vs Voter asymmetry)
        fig_scat2 = px.scatter(
            df_metrics.head(300),
            x="out_degree",
            y="in_degree",
            color="betweenness_centrality",
            hover_name="node_id",
            hover_data=["in_degree", "out_degree", "betweenness_centrality"],
            labels={
                "out_degree": "Out-Degree (Votes Cast)",
                "in_degree": "In-Degree (Votes Received)",
            },
            title="Voter Participation vs Candidate Popularity",
            color_continuous_scale="Plasma",
        )
        fig_scat2.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_scat2, use_container_width=True)

    # Correlation Matrix Table
    pearson_corr, spearman_corr = compute_centrality_correlations(df_metrics)
    col_p, col_s = st.columns(2)
    with col_p:
        st.markdown("#### Pearson Correlation (Linear Co-variance)")
        st.dataframe(
            pearson_corr.style.format("{:.3f}").background_gradient(cmap="Blues"),
            use_container_width=True,
        )
    with col_s:
        st.markdown("#### Spearman Rank Correlation (Monotonic Agreement)")
        st.dataframe(
            spearman_corr.style.format("{:.3f}").background_gradient(cmap="Greens"),
            use_container_width=True,
        )

    st.divider()

    # Individual User Profiler
    st.markdown("### Individual User Ego Profiler")
    st.markdown("Inspect the voting relationships and centrality standings for any specific user.")

    u_col1, u_col2 = st.columns([1, 2])
    with u_col1:
        sample_users = [4037, 15, 2398, 2625, 1297, 1549, 2565, 72]
        selected_user = st.number_input(
            "Enter User Node ID",
            min_value=int(df_metrics["node_id"].min()),
            max_value=int(df_metrics["node_id"].max()),
            value=4037,
            step=1,
        )

        user_row = df_metrics[df_metrics["node_id"] == selected_user]
        if user_row.empty:
            st.warning(f"User {selected_user} not found in network.")
        else:
            u_data = user_row.iloc[0]
            st.success(f"**User {selected_user} Profile Summary**")
            st.markdown(f"- **Community:** Community {int(u_data['community_id'])}")
            st.markdown(f"- **Votes Received (In-Degree):** `{int(u_data['in_degree']):,}`")
            st.markdown(f"- **Votes Cast (Out-Degree):** `{int(u_data['out_degree']):,}`")
            st.markdown(f"- **PageRank:** `{u_data['pagerank']:.6f}` (Rank #{int(u_data['rank_pagerank']):,})")
            st.markdown(
                f"- **Betweenness Centrality:** `{u_data['betweenness_centrality']:.6f}` (Rank #{int(u_data['rank_betweenness']):,})"
            )

    with u_col2:
        if not user_row.empty and selected_user in G:
            voters = list(G.predecessors(selected_user))
            voted_for = list(G.successors(selected_user))

            st.markdown(
                f"#### Direct Voting Ties for User {selected_user} ({len(voters)} voters, {len(voted_for)} nominees)"
            )
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.caption(f"Top 5 Voters who supported User {selected_user}:")
                top_voters = sorted(voters, key=lambda n: G.out_degree(n), reverse=True)[:5]
                st.write(
                    [
                        f"Node {v} (cast {G.out_degree(v)} total votes)"
                        for v in top_voters
                    ]
                    if top_voters
                    else "None"
                )
            with col_v2:
                st.caption(f"Top 5 Nominees User {selected_user} voted for:")
                top_nominees = sorted(voted_for, key=lambda n: G.in_degree(n), reverse=True)[:5]
                st.write(
                    [
                        f"Node {c} (received {G.in_degree(c)} total votes)"
                        for c in top_nominees
                    ]
                    if top_nominees
                    else "None"
                )


# ==============================================================================
# TAB 3: COMMUNITY DETECTION (LOUVAIN)
# ==============================================================================
with tab3:
    st.subheader("Community Detection & Electoral Coalitions (Louvain Algorithm)")
    st.markdown(
        """
        The **Louvain modularity optimization algorithm** uncovers latent voting communities
        and social affinity groups within Wikipedia's election network.
        """
    )
    st.info(
        "⚠️ **SNA Interpretation Caution:** Detected communities represent mathematical partitions "
        "optimizing graph modularity Q on voting ties. They reflect topological clustering within "
        "RfA elections and must NOT be interpreted as real-world off-wiki organizations, ideological factions, "
        "or collusions without external empirical evidence."
    )

    # Theoretical Note Box
    with st.expander("📘 Mathematical Formulation & Undirected Projection Methodology", expanded=False):
        st.markdown(
            r"""
            **1. Newman-Girvan Modularity Objective:**
            $$\mathcal{Q} = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
            where $A_{ij}$ is adjacency matrix, $k_i$ is degree of node $i$, $m$ is total edges,
            and $\delta(c_i, c_j) = 1$ if nodes belong to the same community.

            **2. Undirected Projection Justification:**
            The classical Louvain algorithm assumes symmetric adjacencies. We transform the directed
            voting graph into an undirected weighted projection $G_{undir}$:
            - An undirected edge is created between user $i$ and user $j$ if either $i \to j$ or $j \to i$.
            - Mutual voting relationships ($i \to j$ and $j \to i$) are assigned weight $w=2$;
              unidirectional votes receive weight $w=1$.
            - This preserves mutual reciprocity as a stronger affinity bond in the Louvain optimization.
            """
        )

    # Modularity KPI
    comm_c1, comm_c2, comm_c3 = st.columns(3)
    with comm_c1:
        st.metric(label="Newman-Girvan Modularity (Q)", value=f"{modularity_q:.4f}")
    with comm_c2:
        st.metric(label="Total Communities Detected", value=f"{len(df_community)}")
    with comm_c3:
        largest_size = int(df_community.iloc[0]["node_count"])
        st.metric(
            label="Largest Community Size",
            value=f"{largest_size:,} users",
            delta=f"{(largest_size / G.number_of_nodes()) * 100:.1f}% of network",
        )

    st.divider()

    # Community Size Bar Chart & Table
    col_c_chart, col_c_table = st.columns([1.2, 1])

    with col_c_chart:
        st.markdown("#### Community Size Distribution (Top 12)")
        df_comm_top12 = df_community.head(12)
        fig_comm_bar = px.bar(
            df_comm_top12,
            x="community_id",
            y="node_count",
            text="node_count",
            labels={"community_id": "Community ID", "node_count": "Member Count"},
            title="User Count in Top Detected Communities",
            color="node_count",
            color_continuous_scale="Teal",
        )
        fig_comm_bar.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_comm_bar, use_container_width=True)

    with col_c_table:
        st.markdown("#### Community Leadership & Internal Density")
        st.dataframe(
            df_community[
                [
                    "community_id",
                    "node_count",
                    "internal_edges",
                    "internal_density",
                    "leader_node_id",
                ]
            ].head(12),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # Inter-Community Voting Flow Matrix
    st.markdown("### Inter-Community Directed Voting Flow Heatmap")
    st.caption("Directed votes cast from members of Community (Row) to candidates in Community (Column).")

    matrix_df = get_inter_community_matrix(G, partition, top_n_communities=8)
    fig_heat = px.imshow(
        matrix_df,
        labels=dict(x="Recipient Community (Candidate)", y="Origin Community (Voter)", color="Votes"),
        x=matrix_df.columns,
        y=matrix_df.index,
        color_continuous_scale="Blues",
        text_auto=True,
        aspect="auto",
    )
    fig_heat.update_layout(height=450)
    st.plotly_chart(fig_heat, use_container_width=True)


# ==============================================================================
# TAB 4: INTERACTIVE NETWORK VISUALISATION
# ==============================================================================
with tab4:
    st.subheader("Interactive Network Subgraph Visualisation")
    st.markdown(
        """
        The full dataset contains **7,115 nodes** and **103,689 directed edges**.
        To ensure smooth, responsive 60 FPS exploration and avoid illegible hairballs,
        select a focused subgraph mode below.
        """
    )

    # Subgraph Filtering Controls
    v_col1, v_col2, v_col3, v_col4 = st.columns(4)

    with v_col1:
        subgraph_mode = st.selectbox(
            "Subgraph Sampling Strategy",
            options=[
                "Top Influential Users",
                "Candidate Ego Network",
                "Community Subgraph",
                "High Degree Filter",
            ],
        )

    with v_col2:
        if subgraph_mode == "Top Influential Users":
            sample_k = st.slider("Number of Top Users", min_value=20, max_value=120, value=50, step=10)
            sample_metric = st.selectbox(
                "Metric", options=["in_degree", "pagerank", "betweenness_centrality"]
            )
        elif subgraph_mode == "Candidate Ego Network":
            ego_center = st.number_input("Center Node ID", min_value=1, max_value=9000, value=4037)
            ego_max = st.slider("Max Neighbors", min_value=20, max_value=100, value=50, step=10)
        elif subgraph_mode == "Community Subgraph":
            comm_choice = st.selectbox(
                "Select Community ID", options=list(df_community["community_id"].head(10))
            )
            comm_max = st.slider("Max Nodes in Community", min_value=20, max_value=100, value=50, step=10)
        else:
            deg_thresh = st.slider("Minimum Total Degree", min_value=30, max_value=150, value=70, step=10)

    with v_col3:
        color_choice = st.selectbox(
            "Node Color",
            options=[
                ("community", "Louvain Community ID"),
                ("in_degree", "In-Degree (Votes Received)"),
                ("pagerank", "PageRank Score"),
                ("betweenness_centrality", "Betweenness Centrality"),
            ],
            format_func=lambda x: x[1],
        )[0]

    with v_col4:
        layout_algorithm = st.selectbox(
            "Graph Layout Physics",
            options=["spring", "kamada_kawai", "circular", "spiral"],
            format_func=lambda x: {
                "spring": "Force-Directed (Fruchterman-Reingold)",
                "kamada_kawai": "Path-Distance (Kamada-Kawai)",
                "circular": "Circular Ring",
                "spiral": "Archimedean Spiral",
            }.get(x, x),
        )

    # Extract selected subgraph
    try:
        if subgraph_mode == "Top Influential Users":
            sub_g = extract_top_influencers_subgraph(
                G, df_metrics, top_k=sample_k, metric=sample_metric
            )
            sub_title = f"Top {sample_k} Influential Users Subgraph (Ranked by {sample_metric})"
        elif subgraph_mode == "Candidate Ego Network":
            sub_g = extract_ego_subgraph(
                G, center_node=int(ego_center), radius=1, max_neighbors=ego_max
            )
            sub_title = f"Ego Network for Candidate Node {ego_center} (Voters & Mutual Ties)"
        elif subgraph_mode == "Community Subgraph":
            sub_g = extract_community_subgraph(
                G, partition, community_id=int(comm_choice), max_nodes=comm_max
            )
            sub_title = f"Community {comm_choice} Internal Voting Structure"
        else:
            sub_g = extract_degree_filtered_subgraph(
                G, min_degree=deg_thresh, max_nodes=100
            )
            sub_title = f"High-Activity Core Subgraph (Degree >= {deg_thresh})"

        # Subgraph stats pill
        sub_density = nx.density(sub_g)
        st.caption(
            f"**Rendered Subgraph:** {sub_g.number_of_nodes()} nodes, {sub_g.number_of_edges()} directed edges, "
            f"Density: {sub_density:.4f} | Click, hover, or drag to explore interactive graph nodes."
        )

        # Plot interactive Plotly network
        fig_net = create_plotly_network_figure(
            subgraph=sub_g,
            df_metrics=df_metrics,
            partition=partition,
            layout_type=layout_algorithm,
            color_by=color_choice,
            size_by="in_degree",
            title=sub_title,
        )
        st.plotly_chart(fig_net, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating subgraph: {e}")


# ==============================================================================
# TAB 5: DATA EXPORTS & SNA SOFTWARE COMPATIBILITY
# ==============================================================================
with tab5:
    st.subheader("Data Exports & External SNA Software Integration")
    st.markdown(
        """
        Download complete analysis artifacts in standard Social Network Analysis formats.
        All Pajek exports strictly conform to the Pajek / SocNetV directed `.net` standard.
        """
    )

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("#### 📂 Pajek Network Files (.net)")
        st.markdown(
            "Standard Pajek format with 1-based vertex index mapping and `*Arcs` directed edge declarations."
        )

        pajek_full_file = os.path.join("exports", "wiki_vote_directed.net")
        if os.path.exists(pajek_full_file):
            with open(pajek_full_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download Full Pajek Network (wiki_vote_directed.net)",
                    data=f,
                    file_name="wiki_vote_directed.net",
                    mime="text/plain",
                    help="Full 7,115-node directed voting network for SocNetV, Pajek, or Gephi.",
                )

        pajek_sub_file = os.path.join("exports", "wiki_vote_top100_subgraph.net")
        if os.path.exists(pajek_sub_file):
            with open(pajek_sub_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download Top-100 Subgraph (.net)",
                    data=f,
                    file_name="wiki_vote_top100_subgraph.net",
                    mime="text/plain",
                    help="Lightweight Pajek file for immediate loading in SocNetV.",
                )

    with exp_col2:
        st.markdown("#### 📊 CSV & JSON Analytical Results")
        st.markdown("Complete node-level centralities, community memberships, and edge lists.")

        node_csv_file = os.path.join("exports", "node_centrality_metrics.csv")
        if os.path.exists(node_csv_file):
            with open(node_csv_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download Node Centrality Metrics (CSV)",
                    data=f,
                    file_name="node_centrality_metrics.csv",
                    mime="text/csv",
                )

        comm_csv_file = os.path.join("exports", "community_partitions.csv")
        if os.path.exists(comm_csv_file):
            with open(comm_csv_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download Community Statistics (CSV)",
                    data=f,
                    file_name="community_partitions.csv",
                    mime="text/csv",
                )

        edge_csv_file = os.path.join("exports", "wiki_vote_edges.csv")
        if os.path.exists(edge_csv_file):
            with open(edge_csv_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download Directed Edge List (CSV)",
                    data=f,
                    file_name="wiki_vote_edges.csv",
                    mime="text/csv",
                )

        json_meta_file = os.path.join("exports", "network_overview_metrics.json")
        if os.path.exists(json_meta_file):
            with open(json_meta_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download Network Overview Metrics (JSON)",
                    data=f,
                    file_name="network_overview_metrics.json",
                    mime="application/json",
                )

    st.divider()

    # Step-by-step SocNetV / Pajek Tutorial
    st.markdown("### Step-by-Step Guide: Importing into SocNetV & Pajek")
    with st.expander("📖 Instructions for SocNetV (Social Network Visualizer)", expanded=True):
        st.markdown(
            """
            1. **Launch SocNetV** on your computer.
            2. Go to **File → Load Network...** (or press `Ctrl+O`).
            3. In the file format dialog, choose **Pajek Network (*.net)**.
            4. Select `wiki_vote_top100_subgraph.net` (recommended for instant rendering) or `wiki_vote_directed.net`.
            5. SocNetV automatically parses:
               - The `*Vertices` block (mapping 1-based indices to original Wikipedia user IDs).
               - The `*Arcs` block (directed arrows representing votes from voter to candidate).
            6. In SocNetV, navigate to **Layout → Force-Directed (Spring Embedder)** to compute physical node layout.
            7. Navigate to **Analyze → Centralities** to verify Degree, Betweenness, and Closeness centralities in SocNetV.
            """
        )

    with st.expander("📖 Instructions for Pajek (Large Network Analysis)", expanded=False):
        st.markdown(
            """
            1. **Launch Pajek**.
            2. In the **Network** input row, click the folder icon to browse.
            3. Select `wiki_vote_directed.net`.
            4. Press **Draw → Draw** to visualize the network.
            5. Use **Network → Info → General** to view Pajek's native calculation of density and components.
            """
        )
