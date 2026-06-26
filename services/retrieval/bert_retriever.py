"""
Module Name: bert_retriever.py
Purpose: Online runtime semantic retrieval service using Sentence-BERT and FAISS.
         Supports dynamic topic space pruning using high-speed RAM-level Whitelisting.
"""

import os
import sys
import joblib
import psycopg2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Orchestrate project-level root directories for systemic visibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

STORE_DIR = os.path.join("data", "vector_store")
FAISS_INDEX_PATH = os.path.join(STORE_DIR, "bert_faiss_index.bin")
BERT_IDS_PATH = os.path.join(STORE_DIR, "bert_doc_ids.joblib")

# 🧠 Shared Global In-Memory Singletons for Low-Latency Execution
_bert_model = None
_faiss_index = None
_bert_doc_ids = None


def _init_bert_cache():
    """
    Loads the neural Bi-Encoder and the serialized binary FAISS index into shared RAM
    once during the application boot-cycle to eliminate multi-threaded file system bottlenecks.
    """
    global _bert_model, _faiss_index, _bert_doc_ids

    if _bert_model is None or _faiss_index is None or _bert_doc_ids is None:
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(BERT_IDS_PATH):
            raise FileNotFoundError("CRITICAL: BERT/FAISS index files missing! Run offline training first.")

        print("⏳ Loading BERT Model and FAISS Vector Index into RAM Cache...")
        _bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        _bert_doc_ids = joblib.load(BERT_IDS_PATH)
        print("✅ BERT and FAISS Index loaded into memory successfully!")


def search_bert(query: str, top_k: int = 10, multilingual: bool = False, topic_id: int = None) -> list[dict]:
    """
    Online Ad-hoc Neural Retrieval Engine with Dynamic Post-Filtering Over-sampling.

    IR Core Architecture:
      1. Vector Projection: Maps the runtime unstructured text query into a normalized dense embedding.
      2. Over-sampling Heuristics: If filtered by topic, expands the FAISS candidate footprint to avoid
         dropping semantic matches during strict boolean intersection.
      3. Index Slicing: Scans the high-speed IndexFlatIP binary via optimized C++ vector matrix operations.
      4. Dynamic Intersect: Prunes hits against a relational topic memory template in the RAM heap layer.
      5. Relational Bulk Fetch: Maps dense vector index coordinates to PostgreSQL raw textual strings.
    """
    try:
        _init_bert_cache()

        if not query.strip():
            return []

        # 1. Transform raw query into a normalized continuous embedding vector
        # Setting normalize_embeddings=True brings the vector length to 1.0 (Unit Hypersphere),
        # ensuring the upcoming Inner Product evaluation behaves exactly like Cosine Similarity.
        query_vector = _bert_model.encode([query], normalize_embeddings=True)
        query_vector_faiss = np.array(query_vector).astype('float32')

        conn = psycopg2.connect(
            host="localhost", port=5432, database="ir_search", user="postgres", password="password"
        )
        cursor = conn.cursor()

        # ─────────────────────────────────────────────────────────────────────
        # 🧩 ADVANCED IR PARADIGM: POST-FILTERING VIA CANDIDATE OVER-SAMPLING
        # ─────────────────────────────────────────────────────────────────────
        # Dense vector indices like FAISS IndexFlatIP do not natively support arbitrary
        # relational WHERE clauses (e.g., topic_id = 5) during the index tree lookup.
        #
        # To resolve this architectural limitation, we execute a Post-Filtering strategy:
        #
        # 1. Over-sampling: Instead of asking FAISS for only 'top_k' documents, we cast a wider net
        #    by pulling a larger semantic cluster fraction (`search_limit = max(top_k * 20, 5000)`).
        #    This counters the risk of topical truncation.
        #
        # 2. Boolean Intersection: The expanded candidate pool is swept linearly in high-speed RAM.
        #    Tokens are validated against the PostgreSQL topic lookup hash template (`allowed_ids`).
        #
        # 3. Dynamic Truncation: The moment our whitelisted candidate list meets the requested `top_k`
        #    boundary, the loop exits, preventing redundant processing loops.
        # ─────────────────────────────────────────────────────────────────────
        if topic_id is not None:
            # Pull topical cluster limits from the relational database state
            cursor.execute("SELECT doc_id FROM processed_documents WHERE topic_id = %s;", (topic_id,))
            allowed_ids = set(row[0] for row in cursor.fetchall())

            # Query the FAISS spatial matrix with an expanded search window bounds
            search_limit = max(top_k * 20, 5000)
            scores, indices = _faiss_index.search(query_vector_faiss, search_limit)

            valid_hits = []
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1:
                    doc_id = _bert_doc_ids[idx]
                    if doc_id in allowed_ids:  # Strict structural intersection validation
                        valid_hits.append((doc_id, score))
                        if len(valid_hits) == top_k:  # Target density satisfied, terminate scan
                            break
        else:
            # [Standard Mode]: Scan global unconstrained dense vector index room across 522K boundaries
            scores, indices = _faiss_index.search(query_vector_faiss, top_k)
            valid_hits = []
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1:
                    doc_id = _bert_doc_ids[idx]
                    valid_hits.append((doc_id, score))

        if not valid_hits:
            cursor.close()
            conn.close()
            return []

        # 3. Relational Bulk Fetch (Defeats N+1 database roundtrip penalties)
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

        # Construct immediate O(1) key-value hashmap cache for sequence alignment
        text_map = {row[0]: row[1] for row in db_rows}

        # 4. Formulate standardized response arrays preserving semantic relevance scores
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
        return [{"doc_id": "Error", "score": 0.0, "snippet": f"BERT/FAISS Core Crash: {str(e)}"}]