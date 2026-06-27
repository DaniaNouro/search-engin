import streamlit as st

def render_results(results):
    st.subheader("📄 Results")

    for i, r in enumerate(results, 1):
        st.markdown(f"""
        ### {i}. Document {r['doc_id']}
        **Score:** {r['score']:.4f}

        {r['snippet']}
        ---
        """)
