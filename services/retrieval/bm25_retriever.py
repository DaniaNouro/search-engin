"""
Module Name: bm25_retriever.py
Purpose: Online runtime retrieval service utilizing the pre-trained Okapi BM25 model.
         Performs standard probabilistic ranking across the full corpus footprints.
"""

import os
import sys
import joblib
import psycopg2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.preprocessing.bm25_cleaner import bm25_custom_tokenizer

STORE_DIR = os.path.join("data", "vector_store")
BM25_MODEL_PATH = os.path.join(STORE_DIR, "bm25_model.joblib")
BM25_IDS_PATH = os.path.join(STORE_DIR, "bm25_doc_ids.joblib")

# 🧠 Shared Global In-Memory Singletons
_bm25_model = None
_doc_ids = None


def _init_bm25_cache():
    """
    Loads pre-computed probabilistic arrays into shared cached RAM once to
    eliminate disk read overhead and achieve sub-second execution latencies.
    """
    global _bm25_model, _doc_ids
    if _bm25_model is None or _doc_ids is None:
        if not os.path.exists(BM25_MODEL_PATH) or not os.path.exists(BM25_IDS_PATH):
            raise FileNotFoundError("CRITICAL: BM25 index files missing! Run offline training first.")

        _bm25_model = joblib.load(BM25_MODEL_PATH)
        _doc_ids = joblib.load(BM25_IDS_PATH)


def search_bm25(query: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75) -> list[dict]:
    """
    Online Ad-hoc Probabilistic Retrieval Engine.

    IR Core Architecture:
      1. Dynamic Parameter Injection: Hot-swaps k1 and b hyperparameters per request.
      2. Tokenization Synchronization: Invokes the matched token pipeline to guarantee term overlap alignment.
      3. Statistical Relevance Evaluation: Standard probabilistic ranking across the full corpus volume.
      4. Bulk Context Intersection: Performs array lookups inside PostgreSQL to map hit lists to text snippets.
    """
    try:
        _init_bm25_cache()

        tokenized_query = bm25_custom_tokenizer(query)
        if not tokenized_query:
            return []

        # ── Concept: Dynamic Hyperparameter Tuning ──
        _bm25_model.k1 = k1
        _bm25_model.b = b

        # 1. Compute scores across the full corpus volume
        scores = _bm25_model.get_scores(tokenized_query)

        # 2. Downward sorting and extraction of top K candidates
        top_docs_with_scores = sorted(
            zip(_doc_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Filter out non-actionable matches with zero term overlap (orthogonal records)
        valid_hits = [hit for hit in top_docs_with_scores if hit[1] > 0]
        if not valid_hits:
            return []

        # 3. Relational Bulk Fetch Stage (Prevents database N+1 roundtrip penalties)
        conn = psycopg2.connect(
            host="localhost", port=5432, database="ir_search", user="postgres", password="password"
        )
        cursor = conn.cursor()

        target_ids = [hit[0] for hit in valid_hits]
        score_map = {hit[0]: hit[1] for hit in valid_hits}

        cursor.execute("""
                       SELECT doc_id, raw_text
                       FROM documents
                       WHERE doc_id = ANY (%s);
                       """, (target_ids,))
        db_rows = cursor.fetchall()

        cursor.close()
        conn.close()

        # Build immediate O(1) key-value hash lookup mapping for sequence preservation
        text_map = {row[0]: row[1] for row in db_rows}

        # Build normalized response objects array aligned with UI rendering schemas
        final_results = []
        for doc_id in target_ids:
            if doc_id in text_map:
                final_results.append({
                    "doc_id": doc_id,
                    "score": float(score_map[doc_id]),
                    "snippet": text_map[doc_id]
                })

        return final_results

    except Exception as e:
        return [{"doc_id": "Error", "score": 0.0, "snippet": f"BM25 Core Crash: {str(e)}"}]