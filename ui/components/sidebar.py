import streamlit as st

def render_sidebar():
    st.sidebar.header("⚙️ Settings")

    model = st.sidebar.selectbox(
        "Model",
        ["TF-IDF", "BM25", "Embeddings", "Hybrid"]
    )

    mode = st.sidebar.radio(
        "Mode",
        ["Serial", "Parallel"]
    )

    top_k = st.sidebar.slider("Top K", 1, 50, 10)

    return model, mode, top_k