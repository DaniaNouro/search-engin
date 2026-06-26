import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

# إعداد الصفحة الرئيسية
st.set_page_config(
    page_title="IR Search Engine",
    page_icon="🔍",
    layout="wide"
)

# عرض ترحيبي احترافي
st.title("🔍 Advanced Information Retrieval System")
st.markdown("""
### Welcome to your IR Search Engine project 🚀
This platform provides a unified interface to:
* **Search:** Explore retrieval results using different models (TF-IDF, BM25, BERT).
* **Evaluate:** Analyze system performance using IR metrics (MAP, Precision, Recall).
""")

# لمسة جمالية توضح هيكلية المشروع للمناقشة
with st.expander("ℹ️ About this System Architecture"):
    st.write("""
    This project follows a **Service-Oriented Architecture (SOA)**:
    - **Frontend:** Streamlit (Clean consumer).
    - **API Layer:** FastAPI (Gateway for services).
    - **Backend Engines:** Specialized modules for Preprocessing, Indexing, and Retrieval.
    - **Data Layer:** Centralized storage for vector indices and evaluation metrics.
    """)

st.sidebar.success("Select a page from the sidebar to start!")