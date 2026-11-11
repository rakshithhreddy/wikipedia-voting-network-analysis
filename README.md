# Wikipedia Voting Network Analysis Using Social Network Analytics

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.2%2B-orange.svg)](https://networkx.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, publication-grade Social Network Analytics (SNA) research project analyzing the complete **Wikipedia Request for Adminship (RfA)** voting network from the official **Stanford SNAP** repository.

---

## 📌 Executive Summary & Objective

In Wikipedia's decentralized governance system, editors become administrators via a rigorous peer-review and election process known as **Request for Adminship (RfA)**. The community casts public votes supporting or opposing a candidate.

This project analyzes the entire electoral network using advanced graph theory and Social Network Analysis algorithms to:
1. **Model directed voting relationships** without data distortion or loss of directionality.
2. **Identify influential community members** using In-Degree (popularity), Out-Degree (electoral activity), PageRank (structural prestige), and Betweenness Centrality (social brokerage).
3. **Detect voting coalitions and administrative blocs** via the Louvain community detection algorithm on a documented undirected projection.
4. **Provide interactive visual exploration** through a multi-tab Streamlit dashboard with 2D/3D Plotly network physics.
5. **Ensure interoperability** by exporting strictly compliant Pajek (`.net`) and CSV datasets verified for **SocNetV** and **Pajek**.

---

## 📊 Dataset Specification (Stanford SNAP)

- **Source:** [Stanford Network Analysis Platform (SNAP) Wiki-Vote](https://snap.stanford.edu/data/wiki-Vote.html)
- **Archive URL:** `https://snap.stanford.edu/data/wiki-Vote.txt.gz`
- **Network Type:** Directed, Unweighted (at raw level), Scale-Free
- **Total Nodes ($|V|$):** **7,115 Wikipedia Users**
- **Total Directed Edges ($|E|$):** **103,689 Cast Votes**
- **Directional Semantics:**
  $$\text{Edge } (u \to v) \iff \text{User } u \text{ cast a vote on candidate } v$$
  - **In-Degree ($k_{in}$):** Number of votes received by candidate $v$ (electoral support).
  - **Out-Degree ($k_{out}$):** Number of votes cast by voter $u$ (participation rate).

---

## 🔬 Mathematical Formulations & SNA Methodology

### 1. Macroscopic Topology Metrics

- **Directed Network Density ($D$):**
  $$D = \frac{|E|}{|V|(|V| - 1)} = \frac{103,689}{7,115 \times 7,114} \approx 0.002049$$
  *Interpretation:* Wikipedia voting is sparse ($0.205\%$), as users participate in only a tiny fraction of all historic elections.

- **Arc Reciprocity ($\rho$):**
  $$\rho = \frac{\sum_{i \neq j} A_{ij} A_{ji}}{|E|} \approx 0.0565 \quad (5.65\%)$$
  *Interpretation:* Only $5.65\%$ of voting ties are mutual ($u \to v$ and $v \to u$), highlighting significant hierarchical asymmetry between ordinary voters and administrative candidates.

- **Connected Components:**
  - **Weakly Connected Components (WCC):** $24$ components. The Giant WCC spans **$7,066$ nodes ($99.31\%$)**, confirming that Wikipedia editors form a single globally connected community.
  - **Strongly Connected Components (SCC):** $5,816$ components. The Giant SCC spans **$1,300$ nodes ($18.27\%$)**, representing the cyclical voting core of active editors.

---

### 2. Microscopic Centrality Analysis

| Centrality Metric | Mathematical Definition | SNA Electoral Meaning |
| :--- | :--- | :--- |
| **In-Degree Centrality** | $C_{in}(v) = \frac{\text{deg}_{in}(v)}{N - 1}$ | **Candidate Popularity:** Total volume of votes received. Top node: **Node 4037** ($457$ votes). |
| **Out-Degree Centrality** | $C_{out}(v) = \frac{\text{deg}_{out}(v)}{N - 1}$ | **Electoral Participation:** User engagement in voting. Top voter: **Node 2565** ($893$ votes cast). |
| **PageRank Centrality** | $PR(v) = \frac{1-d}{N} + d \sum_{u \in \text{Pred}(v)} \frac{PR(u)}{\text{deg}_{out}(u)}$ | **Prestige / Reputation:** Endorsement by other high-status, prolific users carries higher weight ($d = 0.85$). |
| **Betweenness Centrality** | $C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$ | **Social Brokerage:** Nodes that bridge distinct voting factions. Documented approximation below. |

#### ⚠️ Documentation of Betweenness Centrality Approximation
Exact betweenness computation requires evaluating all-pairs shortest paths using Brandes' algorithm:
$$\mathcal{O}(|V| \cdot |E|) = 7,115 \times 103,689 \approx 7.38 \times 10^8 \text{ operations}$$
To balance analytical precision with computation speed, we implement the **Brandes & Pich (2007)** unbiased random pivot sampling estimator:
- **Sample Size:** $k = 500$ random pivot vertices with fixed seed (`seed = 42`).
- **Scaling Factor:** Multiplied by $\frac{|V|}{k} = \frac{7,115}{500} = 14.23$.
- **Validation:** Top ranking nodes (Node 1549, Node 2565, Node 15) remain strictly consistent across $k=300$, $k=500$, and full calculations.

---

### 3. Community Detection via Louvain Optimization

#### ⚠️ Documentation of Undirected Projection Methodology
The classical **Louvain algorithm** (Blondel et al., 2008) optimizes Newman-Girvan modularity $Q$:
$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
Because Newman-Girvan modularity requires symmetric adjacency, we project the directed voting network into an undirected weighted graph $G_{undir}$:
1. An edge exists between $u$ and $v$ if $u$ voted for $v$ OR $v$ voted for $u$.
2. **Edge Weighting:**
   - Unidirectional votes ($u \to v$ only) receive weight $w = 1$.
   - Mutual reciprocal votes ($u \to v$ AND $v \to u$) receive weight $w = 2$.
3. **Modularity Score:**
   $$\mathbf{Q = 0.4222} \quad (\text{indicates strong community structure; } Q > 0.3 \text{ is significant})$$
4. **Community Count:** Detected **30 communities**, with the largest containing $2,185$ users ($30.7\%$).

---

## 🗂️ Project Directory Structure

```text
Wikipedia Voting Network Analysis Using Social Network Analytics/
├── app.py                      # Main Streamlit interactive multi-tab dashboard
├── test_pipeline.py            # Comprehensive pipeline verification & test suite
├── requirements.txt            # Pinned Python package dependencies
├── README.md                   # Complete academic documentation and user guide
├── data/                       # Dataset storage and precomputed analytical caches
│   ├── wiki-Vote.txt.gz        # Original Stanford SNAP compressed archive
│   ├── wiki-Vote.txt           # Extracted raw directed edge list
│   └── cache/                  # Fast-loading analytical caches
│       ├── node_metrics.csv    # Centrality values for all 7,115 nodes
│       └── network_summary.json# Macro network metrics
├── exports/                    # Exported standard SNA files (SocNetV / Pajek / CSV)
│   ├── wiki_vote_directed.net  # Valid Pajek format directed network (*Vertices, *Arcs)
│   ├── wiki_vote_top100_subgraph.net # Lightweight Pajek subgraph (for instant SocNetV 3D view)
│   ├── node_centrality_metrics.csv   # Complete node metrics CSV
│   ├── community_partitions.csv      # Louvain community breakdown CSV
│   ├── wiki_vote_edges.csv     # Clean edge list (source_voter, target_candidate)
│   └── network_overview_metrics.json # Macro-level summary metrics
└── src/                        # Modular analysis library
    ├── __init__.py
    ├── data_loader.py          # SNAP download, decompression, DiGraph builder
    ├── network_metrics.py      # Macro SNA metrics, reciprocity, degree distributions
    ├── centrality_analysis.py  # In/out degree, PageRank, Brandes betweenness approx
    ├── community_detection.py  # Louvain on undirected projection & modularity Q
    ├── visualizer.py           # Plotly 2D interactive graphs & Matplotlib static plots
    ├── subgraph_sampler.py     # Subgraph extractors (top influencers, ego networks)
    └── exporters.py            # Strict Pajek (.net) and CSV exporters
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Recommended: Virtual environment (`venv` or `conda`)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Test Suite
Verify that data downloading, graph building, centralities, community detection, and exports operate with zero errors:
```bash
python test_pipeline.py
```
*Expected output: `SUCCESS: ALL PIPELINE TESTS PASSED 100%! (in ~12-14s)`*

### 4. Launch the Interactive Streamlit Dashboard
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Dashboard Features

The dashboard is structured into 5 interactive tabs:

1. **🌐 Network Overview:**
   - 8 KPI cards: Nodes, Edges, Density, Reciprocity, Mean Degree, WCC Coverage, SCC Size, Modularity.
   - Dual log-log degree distribution scatter plots (power-law analysis).
   - User role breakdown (dual users, pure voters, pure candidates).
   - Degree percentile tables (P10 to P100).

2. **🏆 Influential Users:**
   - Top-K ranking tables sorted by In-Degree, PageRank, Betweenness, or Out-Degree.
   - Dynamic comparison bar charts.
   - Cross-metric scatter plots: In-Degree vs PageRank, In-Degree vs Out-Degree.
   - Pearson and Spearman rank correlation matrices.
   - **User Ego Profiler:** Input any node ID (e.g. `4037`, `15`, `2565`) to view its community membership, percentile ranks, top incoming voters, and top candidates voted for.

3. **🧩 Community Detection:**
   - Theoretical overview of Louvain optimization on undirected projection.
   - Size distribution bar chart of the top 12 communities.
   - Community leadership table identifying the primary opinion leader per community.
   - Inter-community voting interaction matrix heatmap.

4. **🕸️ Network Visualisation:**
   - Manageable subgraph extraction to avoid visual hairballing:
     * *Top Influential Users* (mutual edges among top 20-120 leaders).
     * *Candidate Ego Network* (selected user + 1-hop voters and mutual ties).
     * *Community Subgraph* (internal structure of a chosen community).
     * *High Activity Threshold* (nodes with degree $\ge k$).
   - Node coloring: Louvain Community, In-Degree, PageRank, or Betweenness.
   - Node sizing: In-Degree, PageRank, Out-Degree, or Total Degree.
   - Layout physics: Force-Directed (Fruchterman-Reingold), Kamada-Kawai, Circular, or Spiral.
   - Rich hover tooltips with node ID, community, in-degree, out-degree, PageRank, and betweenness.

5. **💾 Data & Exports:**
   - One-click downloads for full Pajek `.net`, top-100 Pajek `.net`, node CSV, community CSV, edge list CSV, and summary JSON.
   - Step-by-step tutorial for importing into SocNetV and Pajek.

---

## 🌐 SocNetV & Pajek Compatibility Assurance

Our exporter strictly follows the official Pajek specification:

```text
*Vertices 7115
1 "3"
2 "4"
...
*Arcs
1 2 1
...
```

### Steps to Open in SocNetV:
1. Open **SocNetV** (Social Network Visualizer).
2. Click **File → Load Network...** (`Ctrl + O`).
3. Select `exports/wiki_vote_top100_subgraph.net` (recommended for smooth rendering) or `exports/wiki_vote_directed.net`.
4. SocNetV parses the `*Vertices` block and `*Arcs` directed relationships.
5. In SocNetV, select **Layout → Force-Directed (Spring Embedder)** to compute physical node coordinates.
6. Select **Analyze → Centralities** to verify Degree, Betweenness, and Closeness centralities.

---

## 📈 Key Empirical Findings

1. **Top Candidate Endorsement (In-Degree):**
   - **Node 4037** is the most endorsed candidate in Wikipedia history, receiving **457 votes**.
   - **Node 15** ranks 2nd with **361 votes**, followed by **Node 2398** (340) and **Node 2625** (331).

2. **Top Voter Activity (Out-Degree):**
   - **Node 2565** is the most prolific voter, casting **893 votes** across elections.
   - Prolific voters serve as key community arbiters who frequently participate in consensus formation.

3. **High Modularity ($Q = 0.4222$):**
   - Confirms that Wikipedia voting is not random but structured into distinct editorial affinity clusters (e.g., topic areas, active time zones, policy factions).

4. **Scale-Free Property:**
   - Both in-degree and out-degree distributions exhibit heavy-tailed linear trends on log-log scales, indicative of preferential attachment where well-known candidates attract disproportionate attention.

---

## 📚 References & Literature

1. **Leskovec, J., Huttenlocher, D., & Kleinberg, J. (2010).** *Predicting positive and negative links in online social networks.* Proceedings of the 19th International Conference on World Wide Web (WWW '10), 641–650.
2. **Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).** *Fast unfolding of communities in large networks.* Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008.
3. **Brandes, U. (2001).** *A faster algorithm for betweenness centrality.* Journal of Mathematical Sociology, 25(2), 163–177.
4. **Brandes, U., & Pich, C. (2007).** *Centrality estimation in large networks.* International Journal of Bifurcation and Chaos, 17(07), 2303–2318.
5. **Page, L., Brin, S., Motwani, R., & Winograd, T. (1999).** *The PageRank citation ranking: Bringing order to the web.* Stanford InfoLab Technical Report.
6. **Batagelj, V., & Mrvar, A. (1998).** *Pajek: A program for large network analysis.* Connections, 21(2), 47–57.
7. **Kalamaras, D. V. (2014).** *SocNetV: Social Networks Visualizer.* Computer software, https://socnetv.org.

---

## 📄 License
This project is open-source under the MIT License.
