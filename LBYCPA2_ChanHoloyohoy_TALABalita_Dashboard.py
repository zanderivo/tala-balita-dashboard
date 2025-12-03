import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from itertools import combinations
import numpy as np
import graphviz
import glob
import gc

# ==========================================
# 1. SETUP & THEME
# ==========================================
st.set_page_config(
    page_title="TALA Balita", 
    page_icon="🧠", 
    layout="wide"
)

st.markdown("""
<style>
    /* Global Background */
    .stApp { background-color: #050505; color: #E0E0E0; }
    
    /* Headers */
    h1, h2, h3 { color: #00FF9D !important; font-family: 'Helvetica Neue', sans-serif; }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background-color: #121212; border-left: 4px solid #00FF9D; padding: 15px;
    }
    
    /* Tabs */
    .stTabs [aria-selected="true"] {
        background-color: #00FF9D !important; color: #000 !important; font-weight: bold;
    }
    
    /* DataFrame selection */
    .stDataFrame { border: 1px solid #333; }
    
    /* About Section Styling */
    .about-card {
        background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #333;
    }
    
    /* Footer Styling */
    .footer-text {
        text-align: center; color: #666; font-size: 0.8em; margin-top: 50px; border-top: 1px solid #333; padding-top: 10px; font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data(ttl="1h", show_spinner="Loading Data...") 
def load_data():
    # Check for split files first
    files = sorted(glob.glob('tala_final_part_*.parquet'))
    
    dfs = []
    if files:
        for f in files:
            chunk = pd.read_parquet(f)
            # Optimize chunk immediately
            for col in ['website', 'category', 'topic_label']:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype('category')
            if 'sentiment' in chunk.columns:
                chunk['sentiment'] = chunk['sentiment'].astype('float32')
            dfs.append(chunk)
        
        df = pd.concat(dfs, ignore_index=True)
        del dfs
    else:
        df = pd.read_parquet('tala_final.parquet')
        for col in ['website', 'category', 'topic_label']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        if 'sentiment' in df.columns:
            df['sentiment'] = df['sentiment'].astype('float32')
            
    gc.collect()
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("🚨 CRITICAL ERROR: 'tala_final.parquet' missing. Run the Pipeline first.")
    st.stop()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def render_footer():
    st.markdown("""
        <div class='footer-text'>
            LBYCPA2 | EQ5 | Chan & Holoyohoy
        </div>
    """, unsafe_allow_html=True)

def get_sankey_data(dataframe):
    df_flow = dataframe.groupby(['website', 'category', 'topic_label']).size().reset_index(name='count')
    df_flow = df_flow[df_flow['count'] > 5] 

    nodes = list(pd.concat([df_flow['website'], df_flow['category'], df_flow['topic_label']]).unique())
    node_map = {n: i for i, n in enumerate(nodes)}
    
    src = [node_map[x] for x in df_flow['website']] + [node_map[x] for x in df_flow['category']]
    tgt = [node_map[x] for x in df_flow['category']] + [node_map[x] for x in df_flow['topic_label']]
    val = df_flow['count'].tolist() + df_flow['count'].tolist()
    
    return nodes, src, tgt, val

def build_network_graph(dataframe, col_name, min_weight=2, top_k_nodes=30):
    G = nx.Graph()
    # Edge Construction
    for items in dataframe[col_name]:
        if isinstance(items, (list, np.ndarray)) and len(items) > 1:
            clean_items = sorted(list(set(items)))
            for u, v in combinations(clean_items, 2):
                if G.has_edge(u, v): G[u][v]['weight'] += 1
                else: G.add_edge(u, v, weight=1)

    # Pruning (Remove weak edges)
    G = nx.Graph([(u,v,d) for u,v,d in G.edges(data=True) if d['weight'] >= min_weight])
    
    # Pruning (Top K Centrality)
    if G.number_of_nodes() > top_k_nodes:
        deg = dict(G.degree(weight='weight'))
        top_nodes = sorted(deg, key=deg.get, reverse=True)[:top_k_nodes]
        G = G.subgraph(top_nodes)
        
    return G

def plot_advanced_network(G, title, color_mode="Single", default_color="#00FF9D"):
    if G.number_of_nodes() < 2:
        st.warning(f"Not enough data to visualize {title}.")
        return

    # LAYOUT ENGINE
    pos = nx.spring_layout(G, k=2.5, seed=42, iterations=50)
    
    # Edge Traces
    edge_x, edge_y = [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()
    
    # Draw Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines', 
        line=dict(width=0.5, color='#444'), hoverinfo='none', showlegend=False
    ))

    # Get all weighted degrees first
    degrees = dict(G.degree(weight='weight'))
    vals = list(degrees.values())
    if not vals: vals = [1]
    min_d, max_d = min(vals), max(vals)
    
    # Helper to scale sizes between 15px and 65px
    def get_size(node_val):
        if max_d == min_d: return 20
        norm = (node_val - min_d) / (max_d - min_d)
        return 15 + (50 * norm)

    # Node Coloring & Drawing Logic
    if color_mode == "Hierarchy":
        # Hierarchy Categories for Places
        categories = {"International": [], "National (NCR)": [], "Regional/Local": []}
        for node in G.nodes():
            if "(International)" in node: categories["International"].append(node)
            elif "(NCR)" in node or "Manila" in node: categories["National (NCR)"].append(node)
            else: categories["Regional/Local"].append(node)
            
        colors = {"International": "#FF00FF", "National (NCR)": "#00E5FF", "Regional/Local": "#FFFF00"}
        
        for cat, nodes in categories.items():
            if not nodes: continue
            nx_subset = [pos[n][0] for n in nodes]
            ny_subset = [pos[n][1] for n in nodes]
            size_subset = [get_size(degrees[n]) for n in nodes]
            
            fig.add_trace(go.Scatter(
                x=nx_subset, y=ny_subset, mode='markers+text',
                text=nodes, textposition="top center",
                name=cat,
                marker=dict(size=size_subset, color=colors[cat], line=dict(width=2, color='white'))
            ))
            
    else: # Single Color Mode (Keywords)
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_txt = list(G.nodes())
        node_sz = [get_size(degrees[n]) for n in G.nodes()]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            text=node_txt, textposition="top center",
            name="Keywords",
            marker=dict(size=node_sz, color=default_color, line=dict(width=1, color='white'))
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="#fff")),
        template="plotly_dark", height=600,
        showlegend=True if color_mode=="Hierarchy" else False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(t=50, b=10, l=10, r=10)
    )
    st.plotly_chart(fig, width="stretch")

# ==========================================
# 4. CONTROLS
# ==========================================
st.sidebar.title("🎛️ TALA Controls")
st.sidebar.caption(f"Loaded: {len(df):,} Articles")

search = st.sidebar.text_input("🔍 Search", "")
min_d, max_d = df['date_clean'].min(), df['date_clean'].max()
date_range = st.sidebar.date_input("📅 Date Range", [min_d, max_d])

st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Leave empty to select ALL")

# 1. Multi-Select Inputs
all_outlets = sorted(df['website'].unique().tolist())
sel_outlet = st.sidebar.multiselect("🌐 Filter Outlets", all_outlets)

all_cats = sorted(df['category'].unique().tolist())
sel_cat = st.sidebar.multiselect("📂 Filter Categories", all_cats)

all_topics = sorted(df['topic_label'].unique().tolist())
sel_topic = st.sidebar.multiselect("🧠 Filter Topics", all_topics)

# 2. Apply Filters
start_d = pd.to_datetime(date_range[0])
end_d = pd.to_datetime(date_range[1])

mask = (df['date_clean'] >= start_d) & (df['date_clean'] <= end_d)
view_df = df.loc[mask].copy()

if sel_outlet: 
    view_df = view_df[view_df['website'].isin(sel_outlet)]
if sel_cat: 
    view_df = view_df[view_df['category'].isin(sel_cat)]
if sel_topic: 
    view_df = view_df[view_df['topic_label'].isin(sel_topic)]

# ==========================================
# 5. DASHBOARD UI
# ==========================================
st.title("📡 TALA Balita")

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Volume", f"{len(view_df):,}")
k2.metric("Topic Clusters", view_df['topic_label'].nunique()) 
k3.metric("Locations Mapped", f"{view_df['locations'].apply(len).sum():,}")
avg_sent = view_df['sentiment'].mean()
k4.metric("Sentiment Index", f"{avg_sent:.3f}", delta="Positive" if avg_sent > 0 else "Negative")

# TABS
tabs = st.tabs(["🗺️ Ecosystem", "🕸️ Networks & Keywords", "📂 Explorer", "ℹ️ About Project"])

# TAB 1: ECOSYSTEM
with tabs[0]:
    st.subheader("Topic Hierarchy")
    tree = view_df.groupby(['category', 'topic_label']).size().reset_index(name='count')
    tree = tree[tree['count']>5]
    fig_tree = px.treemap(tree, path=['category', 'topic_label'], values='count', 
                          color='category', template='plotly_dark')
    st.plotly_chart(fig_tree, width="stretch")
    
    st.divider()
    
    st.subheader("Information Pipeline")
    nodes, src, tgt, val = get_sankey_data(view_df.head(5000))
    if len(nodes) > 0:
        fig_san = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=nodes, color="#00FF9D"),
            link=dict(source=src, target=tgt, value=val, color='rgba(150,150,150,0.3)')
        )])
        fig_san.update_layout(template='plotly_dark', height=500)
        st.plotly_chart(fig_san, width="stretch")
    
    render_footer()

# TAB 2: NETWORKS & KEYWORDS
with tabs[1]:
    
    # SECTION 1: PLACES (With Hierarchy Coloring)
    st.subheader("📍 Geographic Hotspots")
    top_k_places = st.slider("Top Places to Map", 10, 60, 30)
    st.caption("Colors represent geographic hierarchy (International, National, Local). Nodes are sized by frequency.")
    
    G_loc = build_network_graph(view_df.head(5000), 'locations', min_weight=2, top_k_nodes=top_k_places)
    plot_advanced_network(G_loc, "Geographic Network", color_mode="Hierarchy")
    
    st.divider()

    # SECTION 2: KEYWORDS (Network Viz)
    st.subheader("🔗 Keyword Co-occurrence")
    top_k_kw = st.slider("Top Keywords to Connect", 10, 60, 30)
    st.caption("Connects words that appear frequently in the same articles. Nodes sized by frequency.")
    
    all_text = " ".join(view_df['clean_text'].head(1000).tolist())
    words = [w for w in all_text.split() if len(w) > 4]
    
    if words:
        common = [w[0] for w in Counter(words).most_common(top_k_kw)]
        sample = view_df['clean_text'].head(500).apply(lambda t: [w for w in common if w in t])
        G_kw = build_network_graph(pd.DataFrame({'kw': sample}), 'kw', min_weight=2, top_k_nodes=top_k_kw)
        plot_advanced_network(G_kw, "Keyword Network", color_mode="Single", default_color="#FFFF00")
    else:
        st.warning("Not enough text data available for keyword extraction.")

    st.divider()

    # SECTION 3: STATS
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Outlet Volume")
        v_df = view_df['website'].value_counts().reset_index()
        fig_v = px.bar(v_df, x='website', y='count', color='count', template='plotly_dark')
        st.plotly_chart(fig_v, width="stretch")
    with c2:
        st.subheader("Outlet Sentiment")
        s_df = view_df.groupby('website')['sentiment'].mean().reset_index().sort_values('sentiment')
        if not s_df.empty:
            fig_s = px.bar(s_df, y='website', x='sentiment', orientation='h', 
                           color='sentiment', color_continuous_scale='RdBu', template='plotly_dark')
            st.plotly_chart(fig_s, width="stretch")

    render_footer()

# TAB 3: EXPLORER
with tabs[2]:
    st.subheader("📂 Article Explorer")
    
    view_df['id'] = range(len(view_df))
    sel = st.dataframe(
        view_df[['date_clean', 'website', 'title', 'sentiment']],
        column_config={"sentiment": st.column_config.ProgressColumn("Polarity", min_value=-1, max_value=1)},
        on_select="rerun", selection_mode="single-row",
        width="stretch", hide_index=True
    )
    
    if sel.selection.rows:
        row = view_df.iloc[sel.selection.rows[0]]
        st.markdown("---")
        st.markdown(f"### {row['title']}")
        
        c1, c2, c3 = st.columns(3)
        c1.write(f"**Source:** {row['website']}")
        c2.write(f"**Cluster:** {row['topic_label']}")
        c3.write(f"**Date:** {row['date_clean'].date()}")
        
        st.markdown(f"**Locations:** {', '.join(row['locations'])}")
        st.write(f"_{row['clean_text']}..._")
        st.markdown(f"[Read Full Article]({row['url']})")

    render_footer()

# TAB 4: ABOUT & DOCUMENTATION
with tabs[3]:
    st.header("ℹ️ About TALA Balita")
    
    # 1. MISSION ABSTRACT
    st.markdown("""
    <div class="about-card">
        <h3>🚀 Project Mission</h3>
        <p>TALA (Text Analysis & Location Awareness) Balita is an open-source, hybrid AI-powered analytics platform designed to decode the complexity of the Philippine news landscape. 
        By leveraging Natural Language Processing and graph theory, the system transforms unstructured text from over 350,000 articles into structured intelligence, revealing hidden thematic clusters, geographic hotspots, and sentiment trends. 
        This project serves as a transparent, empirical tool for researchers and social scientists, aiming to provide a verifiable methodology for monitoring media coverage and detecting narrative patterns across the nation without relying on opaque proprietary algorithms.</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # 2. USER GUIDE (Split into 3 Sections)
    st.subheader("📖 Dashboard User Guide")
    
    g1, g2, g3 = st.columns(3)
    
    with g1:
        st.markdown("#### 🎛️ 1. Control Panel")
        st.info("""
        **Sidebar Filters**
        The engine allows precise slicing of the dataset.
        
        *   **Search**: Full-text regex search across headlines and bodies.
        *   **Date Range**: Filters the timeline (Default: Full Corpus).
        *   **Outlet**: Select specific sources (e.g., Abante, ABS-CBN).
        *   **Category**: Filter by metadata (e.g., Sports, Metro).
        *   **Topic**: Filter by AI-generated thematic clusters.
        """)

    with g2:
        st.markdown("#### 📊 2. Visualizations")
        st.info("""
        **Ecosystem & Networks**
        
        *   **Treemap**: Hierarchical volume view. (Category → Topic).
        *   **Sankey**: Flow analysis. (Outlet → Category → Topic).
        *   **GeoGraph**: Spatial connections. Pink (Global), Cyan (National), Yellow (Local).
        *   **KeywordGraph**: Semantic co-occurrence. Stronger edges mean words appear together often.
        """)

    with g3:
        st.markdown("#### 📂 3. Data Explorer")
        st.info("""
        **Dataset & Interaction**
        
        *   **Table View**: Click any row to read the full article and see metadata.
        *   **Source**: BalitaNLP Dataset (KenrickLance, 2023).
        *   **Scale**: 351,755 JSON Articles.
        *   **Fields**: Title, Body, Date, Source, URL.
        *   **Split**: 80% Training / 10% Val / 10% Test.
        *   **Fields**: Title, Body, Date, Source, URL.
        *   **Split**: 80% Training / 10% Val / 10% Test.
        """)

    st.divider()

    # 3. SYSTEM ARCHITECTURE (SANDWICH LAYOUT)
    st.subheader("⚙️ System Architecture & Logic")
    
    # --- LEVEL 1: MACRO OVERVIEW ---
    st.markdown("#### 1. Macro Overview: End-to-End Flow")
    macro_dot = """
    digraph Macro {
        rankdir=LR;
        bgcolor="#0e1117";
        node [shape=box, style="filled,rounded", fontname="Helvetica", penwidth=2, margin=0.2];
        # Updated edge: fontcolor set to light grey for readability
        edge [fontname="Helvetica", color="#666666", fontcolor="#CCCCCC", penwidth=2, arrowsize=1];

        # Nodes
        node [fillcolor="#1E1E1E", color="#00FF9D", fontcolor="white"];
        
        Input [label="Raw Input\n(BalitaNLP JSON)"];
        Engine [label="Processing Engine\n(TALA Backend .ipynb)"];
        Storage [label="Data Warehouse\n(Parquet File)"];
        App [label="Analytics App\n(Streamlit Dashboard .py)"];

        # Edges
        Input -> Engine [label=" Ingest"];
        Engine -> Storage [label=" Transform & Load"];
        Storage -> App [label=" Serve"];
    }
    """
    st.graphviz_chart(macro_dot, width="stretch")

    # --- LEVEL 2: BACKEND PIPELINE ---
    st.markdown("#### 2. Backend: Data Processing Pipeline (.ipynb)")
    backend_dot = """
    digraph Backend {
        rankdir=LR;
        bgcolor="#0e1117";
        node [shape=record, style="filled", fontname="Courier", fontsize=10, penwidth=1.5];
        edge [arrowhead="vee", color="#888", penwidth=1.5];
        
        # Nodes
        node [fillcolor="#2D2D2D", color="#00FF9D", fontcolor="white"];

        subgraph cluster_ingest {
            label = "Phase 1: Ingestion";
            fontcolor="#888"; color="#444"; style=dashed;
            JSON [label="{Raw Files|Train/Test/Val}"];
            DynArray [label="{Dynamic Array|List[Dict] Staging}"];
        }

        subgraph cluster_etl {
            label = "Phase 2: Transformation";
            fontcolor="#888"; color="#444"; style=dashed;
            Clean [label="{Cleaning|Regex & Stopwords}"];
            Vector [label="{Vectorization|TF-IDF Matrix}"];
            NMF [label="{AI Modeling|NMF Topic Clusters}"];
            HashMap [label="{Entity Res|Geo-Hash Lookup}"];
        }

        subgraph cluster_out {
            label = "Phase 3: Export";
            fontcolor="#888"; color="#444"; style=dashed;
            Export [label="{Output|tala_final.parquet}"];
        }

        # Flow
        JSON -> DynArray;
        DynArray -> Clean;
        Clean -> Vector;
        Clean -> HashMap;
        Vector -> NMF;
        NMF -> Export;
        HashMap -> Export;
    }
    """
    st.graphviz_chart(backend_dot, width="stretch")

    # --- LEVEL 3: FRONTEND PIPELINE ---
    st.markdown("#### 3. Frontend: Visualization Logic (.py)")
    frontend_dot = """
    digraph Frontend {
        rankdir=LR;
        bgcolor="#0e1117";
        node [shape=note, style="filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [arrowhead="vee", color="#888", penwidth=1.5];

        # Nodes
        node [fillcolor="#2D2D2D", color="#00FF9D", fontcolor="white"];

        Load [label="Load Data\n(@st.cache_data)"];
        Pandas [label="Pandas DataFrame\n(In-Memory)"];
        
        subgraph cluster_interact {
            label = "User Interaction";
            fontcolor="#888"; color="#444"; style=dashed;
            Filters [label="Sidebar Controls\n(Date, Outlet, Topic)"];
            Mask [label="Boolean Masking\n(df.loc[mask])"];
        }

        subgraph cluster_compute {
            label = "Graph Computation";
            fontcolor="#888"; color="#444"; style=dashed;
            NX [label="NetworkX Build\n(Nodes & Edges)"];
            Tree [label="Tree Grouping\n(Hierarchy)"];
        }

        Render [label="Plotly Render\n(Canvas Draw)"];

        # Flow
        Load -> Pandas;
        Pandas -> Filters;
        Filters -> Mask;
        Mask -> NX;
        Mask -> Tree;
        NX -> Render;
        Tree -> Render;
    }
    """
    st.graphviz_chart(frontend_dot, use_container_width=True)

    render_footer()