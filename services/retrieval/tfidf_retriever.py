"""
Module Name: tfidf_retriever.py
Purpose: Online retrieval service for TF-IDF Vector Space Model.
         Loads pre-computed joblib indices into memory (cached), processes runtime queries,
         computes Cosine Similarity, and fetches original text snippets from PostgreSQL efficiently.
"""

import os
import joblib
import numpy as np
import psycopg2

# Centralized data store absolute path orchestration
VECTOR_STORE_DIR = os.path.join("data", "vector_store")
VECTORIZER_PATH = os.path.join(VECTOR_STORE_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(VECTOR_STORE_DIR, "tfidf_matrix.joblib")
DOC_IDS_PATH = os.path.join(VECTOR_STORE_DIR, "doc_ids.joblib")

# 🧠 In-Memory Singletons / Global Index Caching
# IR Concept: Low-latency Serving — preserves sparse matrices inside shared RAM
# to mitigate disk I/O bottlenecks across multi-threaded web requests.
_vectorizer = None
_tfidf_matrix = None
_doc_ids = None


def _lazy_load_indices() -> bool:
    """
    Implements a thread-safe style Lazy Loading strategy to pull trained VSM
    primitives into RAM exactly once during the server lifecycle.
    """
    global _vectorizer, _tfidf_matrix, _doc_ids

    if _vectorizer is None or _tfidf_matrix is None or _doc_ids is None:
        if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MATRIX_PATH) and os.path.exists(DOC_IDS_PATH)):
            return False

        print("📥 [MEMORY LOADING] Loading massive 522K TF-IDF models into cached RAM for fast online searching...")
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _tfidf_matrix = joblib.load(MATRIX_PATH)
        _doc_ids = joblib.load(DOC_IDS_PATH)
        print("🚀 [MEMORY LOADED] Core TF-IDF models ready in RAM.")

    return True


def search_tfidf(query: str, top_k: int = 10) -> list[dict]:
    """
    Online Ad-hoc Retrieval Engine using Vector Space Model scoring.

    IR Pipeline Architecture:
      1. Query Projection: Vectorizes the dynamic query string using the pre-compiled VSM dictionary.
      2. Similarity Estimation: Computes spatial Cosine Similarity between the query vector and corpus matrix.
      3. Index Slicing: Extracted indices are sorted downwards via numpy priority arg-sorting.
      4. Dynamic Bulk Pruning: Collects non-zero relevance scores to avoid empty vocabulary overlaps.
      5. Relational Bulk Fetch: Uses PostgreSQL arrays `ANY (%s)` to isolate textual snippets in a single database hop.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # 1. Verify index integrity and invoke global caching
    if not _lazy_load_indices():
        return [{
            "doc_id": "Error",
            "score": 0.0,
            "snippet": "Index files not found. Please execute the offline training pipeline first!"
        }]

    try:
        # 2. Transform the dynamic raw text query into a high-dimensional spatial vector
        query_vector = _vectorizer.transform([query])

        # ─────────────────────────────────────────────────────────────────────
        # 📐 MATHEMATICAL FOUNDATION: COSINE SIMILARITY IN VSM
        # ─────────────────────────────────────────────────────────────────────
        # In Information Retrieval, semantic relevance is determined by the geometric
        # angle between the Query Vector 'q' and Document Vector 'd' in a multi-dimensional
        # space, completely ignoring document length scale factors.
        #
        # Formula:
        #    Cosine_Similarity(q, d) = (q • d) / (||q|| * ||d||)
        #
        # Deconstructed:
        #    Numerator (Dot Product):   sum( q_i * d_i )
        #    Denominator (L2 Norms):    sqrt( sum(q_i^2) ) * sqrt( sum(d_i^2) )
        #
        # Value Space:
        #    Since TF-IDF weights are non-negative, the score is bounded strictly in [0.0, 1.0].
        #    - Score = 1.0 -> Vectors point in identical directions (High lexical mapping).
        #    - Score = 0.0 -> Vectors are orthogonal (Zero shared vocabulary terms).
        # ─────────────────────────────────────────────────────────────────────
        similarity_scores = cosine_similarity(query_vector, _tfidf_matrix).flatten()

        # 4. Sort indices in descending order using fast multi-axis numpy sorting
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    except Exception as e:
        return [{"doc_id": "Error", "score": 0.0, "snippet": f"Failed to process query math: {str(e)}"}]

    # 5. Prune and capture relevant documents where score strictly exceeds 0.0 boundary
    matched_docs = []
    for idx in top_indices:
        score = similarity_scores[idx]
        if score > 0.0:
            matched_docs.append({
                "doc_id": _doc_ids[idx],
                "score": float(score)
            })

    if not matched_docs:
        return []

    # 6. Database Bulk Intersection (Optimized to defeat N+1 query performance anti-patterns)
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="ir_search",
            user="postgres",
            password="password"
        )
        cursor = conn.cursor()

        # Build centralized array payload for relational indexing fetch
        target_ids = [doc["doc_id"] for doc in matched_docs]

        # Using ANY (%s) native matrix match which is drastically faster than standard IN clauses
        cursor.execute("""
                       SELECT doc_id, raw_text
                       FROM documents
                       WHERE doc_id = ANY (%s);
                       """, (target_ids,))

        # Map dynamic results directly into an O(1) Python Hashmap lookup dictionary
        db_results = {str(row[0]): row[1] for row in cursor.fetchall()}

        cursor.close()
        conn.close()

        # 7. Synthesize final response structures, enforcing absolute alignment with VSM similarity rank
        final_results = []
        for doc in matched_docs:
            d_id = str(doc["doc_id"])
            final_results.append({
                "doc_id": d_id,
                "score": doc["score"],
                "snippet": db_results.get(d_id, "Document context missing in database record.")
            })

        return final_results

    except Exception as e:
        return [{"doc_id": "Error", "score": 0.0, "snippet": f"Database Bulk Interaction failed: {str(e)}"}]