# 📡 TALA Balita: Text Analysis & Location Awareness

**TALA Balita** is an open-source, hybrid AI-powered analytics platform designed to decode the complexity of the Philippine news landscape. By leveraging Natural Language Processing (NLP) and graph theory, the system transforms unstructured text from over 350,000 articles into structured intelligence, revealing hidden thematic clusters, geographic hotspots, and sentiment trends.

This project serves as a transparent, empirical tool for researchers and social scientists, aiming to provide a verifiable methodology for monitoring media coverage and detecting narrative patterns across the nation without relying on opaque proprietary algorithms.

## 🚀 Key Features

### 🎛️ Interactive Control Panel
*   **Precise Filtering**: Slice the dataset by Date Range, Media Outlet, Category, or Topic Cluster.
*   **Full-Text Search**: Regex-enabled search across headlines and article bodies.

### 📊 Advanced Visualizations
*   **Topic Hierarchy**: Treemaps showing the distribution of topics within categories.
*   **Information Pipeline**: Sankey diagrams visualizing the flow from Outlet → Category → Topic.
*   **Geographic Network**: A spatial graph connecting locations mentioned in articles, colored by hierarchy (International, National, Local).
*   **Keyword Co-occurrence**: A semantic network showing how words appear together, revealing narrative structures.

### 📂 Data Explorer
*   **Deep Dive**: Clickable table view to read full articles and inspect metadata.
*   **Sentiment Analysis**: Integrated polarity scores for every article.

## 🛠️ Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Data Processing**: Pandas, NumPy
*   **Visualization**: Plotly, NetworkX, Graphviz
*   **Data Format**: Parquet (Split for performance)

## 💻 Running Locally

1.  **Clone the repository**
    ```bash
    git clone https://github.com/zanderivo/tala-balita-dashboard.git
    cd tala-balita-dashboard
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**
    ```bash
    streamlit run LBYCPA2_ChanHoloyohoy_TALABalita_Dashboard.py
    ```

## ℹ️ Project Info

*   **Course**: LBYCPA2
*   **Team**: EQ5 - Chan & Holoyohoy
*   **Dataset**: BalitaNLP (351,755 Articles)
