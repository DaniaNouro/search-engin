import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Information Retrieval Evaluation Dashboard")

# ==========================================================
# Load Evaluation Files
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RESULT_FILES = [
    BASE_DIR / "data/evaluation_results/eval_tf-idf.json",
    BASE_DIR / "data/evaluation_results/eval_bm25.json",
    BASE_DIR / "data/evaluation_results/eval_bert.json",
    BASE_DIR / "data/evaluation_results/eval_hybrid_parallel.json",
]

rows = []

for file_path in RESULT_FILES:
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

if not rows:
    st.error("No evaluation files found.")
    st.stop()

df = pd.DataFrame(rows)

# ==========================================================
# Metric Explanations
# ==========================================================

st.header("📖 Evaluation Metrics Explanation")

with st.expander("MAP (Mean Average Precision)", expanded=False):
    st.markdown("""
### What is MAP?

MAP measures how well relevant documents are ranked.

### High MAP
✅ Relevant documents appear early.

### Low MAP
❌ Relevant documents appear late.

### Example

Query:

`machine learning`

Good ranking:

1. Relevant
2. Relevant
3. Relevant

Bad ranking:

1. Irrelevant
2. Irrelevant
3. Relevant
""")

with st.expander("Precision@10"):
    st.markdown("""
### What is Precision@10?

Percentage of relevant documents in the first 10 retrieved results.

### Formula

Precision@10 = Relevant Results in Top 10 / 10

### Example

3 relevant documents among top 10:

Precision@10 = 3 / 10 = 0.30

### Interpretation

High:
✅ Accurate results

Low:
❌ Many irrelevant results
""")

with st.expander("Recall@10"):
    st.markdown("""
### What is Recall@10?

Measures how many relevant documents were found.

### Formula

Recall = Retrieved Relevant / Total Relevant

### Example

20 relevant documents exist.

System found 15.

Recall = 15 / 20 = 0.75

### Interpretation

High:
✅ Finds most relevant documents

Low:
❌ Misses important documents
""")

with st.expander("nDCG"):
    st.markdown("""
### What is nDCG?

Measures ranking quality.

Not all relevant documents have equal importance.

### High nDCG

✅ Most important results appear first.

### Low nDCG

❌ Important documents appear late.
""")

with st.expander("Latency"):
    st.markdown("""
### What is Latency?

Time needed to answer a query.

### High Latency

❌ Slow system

### Low Latency

✅ Fast system
""")

# ==========================================================
# Best Models
# ==========================================================

st.header("🏆 Best Performing Models")

col1, col2, col3 = st.columns(3)

best_map = df.loc[df["MAP"].idxmax()]
best_ndcg = df.loc[df["nDCG"].idxmax()]
fastest = df.loc[df["Avg_Latency_ms"].idxmin()]

with col1:
    st.metric(
        "Highest MAP",
        best_map["model"],
        f"{best_map['MAP']:.3f}"
    )

with col2:
    st.metric(
        "Highest nDCG",
        best_ndcg["model"],
        f"{best_ndcg['nDCG']:.3f}"
    )

with col3:
    st.metric(
        "Fastest Model",
        fastest["model"],
        f"{fastest['Avg_Latency_ms']:.1f} ms"
    )

# ==========================================================
# Comparison Table
# ==========================================================

st.header("📋 Comparison Table")

display_df = df[
    [
        "model",
        "MAP",
        "Precision@10",
        "Recall@10",
        "nDCG",
        "Avg_Latency_ms",
    ]
].copy()

st.dataframe(
    display_df,
    use_container_width=True,
)

# ==========================================================
# Metric Comparison Charts
# ==========================================================

st.header("📈 Metrics Comparison")

metrics = [
    "MAP",
    "Precision@10",
    "Recall@10",
    "nDCG",
]

for metric in metrics:

    fig = px.bar(
        df,
        x="model",
        y=metric,
        title=f"{metric} Comparison",
        text_auto=".3f",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

# ==========================================================
# Latency Chart
# ==========================================================

fig_latency = px.bar(
    df,
    x="model",
    y="Avg_Latency_ms",
    title="Latency Comparison (ms)",
    text_auto=".0f",
)

st.plotly_chart(
    fig_latency,
    use_container_width=True,
)

# ==========================================================
# Radar Chart
# ==========================================================

st.header("🕸️ Radar Comparison")

radar_df = df.copy()

radar_df["LatencyScore"] = (
    radar_df["Avg_Latency_ms"].max()
    - radar_df["Avg_Latency_ms"]
)

categories = [
    "MAP",
    "Precision@10",
    "Recall@10",
    "nDCG",
    "LatencyScore",
]

fig = go.Figure()

for _, row in radar_df.iterrows():

    values = [
        row["MAP"],
        row["Precision@10"],
        row["Recall@10"],
        row["nDCG"],
        row["LatencyScore"],
    ]

    values.append(values[0])

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["model"],
        )
    )

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
        )
    ),
    showlegend=True,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

# ==========================================================
# Conclusion
# ==========================================================

st.header("📝 Conclusion")

best_overall = df.loc[
    (
        df["MAP"]
        + df["nDCG"]
        + df["Recall@10"]
        + df["Precision@10"]
    ).idxmax()
]

st.success(
    f"""
Best overall retrieval model:
{best_overall["model"]}

MAP = {best_overall["MAP"]:.3f}

Precision@10 = {best_overall["Precision@10"]:.3f}

Recall@10 = {best_overall["Recall@10"]:.3f}

nDCG = {best_overall["nDCG"]:.3f}
"""
)