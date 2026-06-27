# import os
import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from components.results import render_results
from ui.services.api_client import fetch_search_results
from services.retrieval.history_service import save_query
from services.preprocessing.query_refiner import (
    refine_query,
    suggest_queries_from_history,
    formulate_query_with_context,
)

st.title("🔍 Search Engine")

USER_ID = "default_user"

# ------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# معالجة أي قيمة جديدة قبل إنشاء الـ widget
if st.session_state.pending_query is not None:
    st.session_state.query_input = st.session_state.pending_query
    st.session_state.pending_query = None

# ------------------------------------------------------------------
# Query Input
# ------------------------------------------------------------------
st.text_input(
    "Enter your query",
    key="query_input",
    placeholder="What are you looking for today?",
)

current_query = st.session_state.query_input

# ------------------------------------------------------------------
# Query Assistance
# ------------------------------------------------------------------
if current_query.strip():

    corrected_query = refine_query(current_query)

    if corrected_query.strip().lower() != current_query.strip().lower():
        st.markdown(f"💡 **Did you mean:** `{corrected_query}` ?")

        if st.button("Yes, use corrected query", key="spell_fix_btn"):
            st.session_state.pending_query = corrected_query
            st.rerun()

    smart_completions = suggest_queries_from_history(
        current_query,
        user_id=USER_ID,
        limit=5,
    )

    if smart_completions:
        st.markdown("🔮 **Search Suggestions (from your history):**")

        for idx, comp_query in enumerate(smart_completions):
            if st.button(
                f"🔍 {comp_query}",
                key=f"comp_{idx}_{hash(comp_query)}",
                use_container_width=True,
            ):
                st.session_state.pending_query = comp_query
                st.rerun()

# ------------------------------------------------------------------
# Retrieval Model Selection
# ------------------------------------------------------------------
model = st.selectbox(
    "Retrieval Model",
    ["TF-IDF", "BM25", "BERT", "Hybrid"],
)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.markdown("### 🧩 Extra Features")

enable_multilingual = st.sidebar.checkbox(
    "Enable Multilingual Search",
    help="Enable cross-language retrieval (AR/EN)",
)

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
with st.expander(
    "🛠️ Advanced Model Configurations (Hyperparameters)",
    expanded=True,
):

    k1, b = 1.5, 0.75
    alpha, beta = 0.5, 0.5
    hybrid_mode = "parallel"

    if model == "BM25":
        st.markdown("**BM25 Tuning:**")

        col1, col2 = st.columns(2)

        with col1:
            k1 = st.slider(
                "Term Frequency Saturation (k1)",
                1.2,
                2.0,
                1.5,
                step=0.1,
            )

        with col2:
            b = st.slider(
                "Document Length Normalization (b)",
                0.5,
                1.0,
                0.75,
                step=0.05,
            )

    elif model == "Hybrid":

        st.markdown("**Hybrid Mode Execution:**")

        hybrid_mode = st.radio(
            "Execution Flow",
            ["Serial", "Parallel"],
            help="Serial: Cascaded filtering. Parallel: Score Fusion via normalized weights.",
        ).lower()

        st.markdown("**Score Fusion Weights:**")

        col1, col2 = st.columns(2)

        with col1:
            alpha = st.slider(
                "Lexical Weight (α - BM25/TF-IDF)",
                0.0,
                1.0,
                0.5,
                step=0.1,
            )

        with col2:
            beta = st.slider(
                "Semantic Weight (β - BERT)",
                0.0,
                1.0,
                1.0 - alpha,
                step=0.1,
            )

            if alpha + beta != 1.0:
                st.caption(
                    f"⚠️ Note: Current sum is {alpha + beta:.1f}. "
                    "Best results usually yield 1.0."
                )

top_k = st.slider("Top K results", 1, 20, 10)

# ------------------------------------------------------------------
# Search Execution
# ------------------------------------------------------------------
if st.button("Search"):

    if not current_query.strip():
        st.warning("Please enter a valid query.")

    else:

        refined_query = refine_query(current_query.strip())

        formulation = formulate_query_with_context(
            refined_query,
            user_id=USER_ID,
        )

        retrieval_query = formulation.retrieval_query

        if formulation.was_enriched and formulation.context_keyword:
            st.caption(
                f"🎯 Contextual retrieval active — profile term "
                f"**`{formulation.context_keyword}`** applied to boost relevance."
            )

        save_query(
            user_id=USER_ID,
            query_text=formulation.display_query,
            model_used=model,
        )

        with st.spinner(f"Searching through {model} API Service..."):

            search_params = {
                "query": retrieval_query,
                "model": model,
                "top_k": top_k,
                "k1": k1,
                "b": b,
                "hybrid_mode": hybrid_mode,
                "alpha": alpha,
                "beta": beta,
                "multilingual": enable_multilingual,
            }

            results = fetch_search_results(**search_params)

            if results and results[0]["doc_id"] != "Error":
                render_results(results)

            elif results and results[0]["doc_id"] == "Error":
                st.error(results[0]["snippet"])

            else:
                st.info("No matching documents found.")