# import streamlit as st
# import pandas as pd
# from ui.services.api_client import fetch_evaluation_metrics
#
# st.title("📈 Model Comparison & Evaluation")
#
# # 1. تهيئة ذاكرة الجلسة لتخزين النتائج المقارنة
# if 'comparison_df' not in st.session_state:
#     st.session_state.comparison_df = pd.DataFrame()
#
# target_model = st.selectbox("Select Model to Evaluate", ["TF-IDF", "BM25", "BERT"])
# num_queries = st.slider("Number of Queries for Evaluation Sample", 10, 200, 50)
# eval_top_k = st.slider("Top K for Metrics", 5, 20, 10)
#
# if st.button("📊 Run System Evaluation"):
#     with st.spinner(f"Evaluating {target_model} against ground truth Qrels..."):
#         # جلب المقاييس (MAP, Recall, Precision, Latency) من الباكيند
#         metrics = fetch_evaluation_metrics(model=target_model, limit_queries=num_queries, top_k=eval_top_k)
#
#         if "Error" in metrics:
#             st.error(metrics["Error"])
#         else:
#             # إضافة اسم الموديل للنتائج
#             metrics['Model'] = target_model
#
#             # 2. تحديث جدول المقارنة في الـ session_state (مع منع تكرار نفس الموديل)
#             new_df = pd.DataFrame([metrics])
#             if not st.session_state.comparison_df.empty:
#                 # حذف السجل القديم للموديل إذا كان موجوداً ليتم تحديثه بالأرقام الجديدة
#                 st.session_state.comparison_df = st.session_state.comparison_df[
#                     st.session_state.comparison_df['Model'] != target_model]
#
#             st.session_state.comparison_df = pd.concat([st.session_state.comparison_df, new_df], ignore_index=True)
#             st.success(f"Results for {target_model} added to comparison table successfully!")
#
# # 3. عرض جدول المقارنة إذا كان يحتوي على بيانات
# if not st.session_state.comparison_df.empty:
#     st.subheader("📊 Comparative Benchmark Matrix")
#
#     # تنسيق الأعمدة ليظهر اسم الموديل أولاً لسهولة القراءة
#     cols = ['Model'] + [c for c in st.session_state.comparison_df.columns if c != 'Model']
#     st.table(st.session_state.comparison_df[cols])
#
#     if st.button("🗑️ Clear Comparison Data"):
#         st.session_state.comparison_df = pd.DataFrame()
#         st.rerun()
#
# # 4. التوصيف العلمي للموديلات الثلاثة أمام لجنة التحكيم
# st.info("""
# 💡 **Academic Comparative Analysis Insight (For Presentation):**
# * **TF-IDF:** Base baseline. Fast, but misses document length context and document frequency tuning.
# * **BM25 (Lexical Champion):** Optimizes term frequency saturation and document length normalization. Expect high speed with very solid MAP on exact matches.
# * **BERT (Semantic Champion):** Captures full deep contextual and paraphrasing meanings (e.g., matching synonyms). Expect the highest **MAP & Recall** on conversational queries, with a natural microsecond latency trade-off due to neural vector similarity calculations via FAISS.
# """)

import streamlit as st
import pandas as pd
import json
import os
from ui.services.api_client import fetch_evaluation_metrics

st.title("📈 Model Evaluation & Comparison")

# ── helper ──────────────────────────────────────────────────────────────
def _update_comparison(metrics: dict, model_name: str):
    metrics['Model'] = model_name
    new_df = pd.DataFrame([metrics])
    if not st.session_state.comparison_df.empty:
        st.session_state.comparison_df = st.session_state.comparison_df[
            st.session_state.comparison_df['Model'] != model_name
        ]
    st.session_state.comparison_df = pd.concat(
        [st.session_state.comparison_df, new_df], ignore_index=True
    )

# ── session state ────────────────────────────────────────────────────────
if 'comparison_df' not in st.session_state:
    st.session_state.comparison_df = pd.DataFrame()

# ── controls ─────────────────────────────────────────────────────────────
target_model = st.selectbox(
    "Select Model to Evaluate",
    ["TF-IDF", "BM25", "BERT", "HYBRID_PARALLEL", "HYBRID_SERIAL"]
)

eval_top_k = st.slider("Top K for Metrics", 5, 20, 10)

if target_model in ["BERT", "HYBRID_PARALLEL", "HYBRID_SERIAL"]:
    st.warning("⚠️ This model on 10,000 queries may take a long time. Consider loading saved results instead.")

# ── buttons ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Run Full Evaluation (All Queries)"):
        with st.spinner(f"Evaluating {target_model} on ALL qrels queries..."):
            metrics = fetch_evaluation_metrics(
                model=target_model,
                limit_queries=10000,
                top_k=eval_top_k
            )
            if "Error" in metrics:
                st.error(metrics["Error"])
            else:
                st.success(
                    f"✅ Done! Used **{metrics.get('total_queries_used', '?')}** queries"
                )
                _update_comparison(metrics, target_model)

with col2:
    if st.button("📂 Load Saved Results"):
        path = os.path.join(
            "data", "evaluation_results",
            f"eval_{target_model.lower()}.json"
        )
        if os.path.exists(path):
            with open(path) as f:
                metrics = json.load(f)
            st.success(f"✅ Loaded saved results for **{target_model}**")
            _update_comparison(metrics, target_model)
        else:
            st.warning("No saved results found. Run evaluation first.")

# ── comparison table ──────────────────────────────────────────────────────
if not st.session_state.comparison_df.empty:
    st.subheader("📊 Comparative Benchmark Matrix")

    priority_cols = [
        'Model',
        'total_queries_used',
        'MAP',
        'nDCG',
        f'Precision@{eval_top_k}',
        f'Recall@{eval_top_k}',
        'Avg_Latency_ms'
    ]
    cols = [c for c in priority_cols
            if c in st.session_state.comparison_df.columns]

    st.dataframe(
        st.session_state.comparison_df[cols],
        use_container_width=True
    )

    if st.button("🗑️ Clear Comparison Data"):
        st.session_state.comparison_df = pd.DataFrame()
        st.rerun()

# ── academic insight ──────────────────────────────────────────────────────
st.info("""
💡 **Academic Comparative Analysis:**
- **TF-IDF:** Base baseline. Fast but ignores term frequency saturation and document length.
- **BM25:** Probabilistic model. Optimizes TF saturation (k1) and length normalization (b). Strong on exact matches.
- **BERT:** Semantic model. Captures contextual meaning via dense vectors. Best MAP on conversational queries.
- **Hybrid Parallel:** Fuses BM25 + BERT scores with weighted combination. Wider coverage.
- **Hybrid Serial:** BM25 filters candidates → BERT re-ranks. Faster than full BERT scan.
""")