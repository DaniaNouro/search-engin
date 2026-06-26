"""
Module Name: multilingual_retriever.py
Purpose: Extra Feature - Cross-Language Semantic Search (Arabic/English)
         using paraphrase-multilingual-MiniLM-L12-v2 and a dedicated FAISS Index.
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
MULTI_FAISS_INDEX_PATH = os.path.join(STORE_DIR, "multilingual_faiss_index.bin")
MULTI_IDS_PATH = os.path.join(STORE_DIR, "multilingual_doc_ids.joblib")

# 🧠 Isolated In-Memory Singletons for the Multilingual Expansion Feature
# Keeps memory states unentangled from the primary monolingual BERT channels
_multi_model = None
_multi_faiss_index = None
_multi_doc_ids = None


def _init_multilingual_cache():
    """
    Loads cross-lingual transformer parameters and pre-computed vector matrices into shared RAM
    once to secure fast execution paths and eliminate disk read spikes.
    """
    global _multi_model, _multi_faiss_index, _multi_doc_ids

    if _multi_model is None or _multi_faiss_index is None or _multi_doc_ids is None:
        if not os.path.exists(MULTI_FAISS_INDEX_PATH) or not os.path.exists(MULTI_IDS_PATH):
            raise FileNotFoundError("EXTRA FEATURE ERROR: Multilingual FAISS index files missing! Run offline multilingual training script first.")

        print("⏳ Loading Multilingual BERT Model and Multi-FAISS Index into RAM Cache...")
        # Instantiates the global massively multilingual topology mapped to a shared spatial domain
        _multi_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        _multi_faiss_index = faiss.read_index(MULTI_FAISS_INDEX_PATH)
        _multi_doc_ids = joblib.load(MULTI_IDS_PATH)
        print("✅ Multilingual System loaded into memory successfully!")


def search_multilingual(query: str, top_k: int = 10) -> list[dict]:
    """
    Online Ad-hoc Cross-Language Semantic Search Service.

    IR Execution Pipeline:
      1. Cross-Lingual Projection: Encodes runtime query strings (e.g., Arabic) directly into
         the shared dense vector room, applying strict L2 normalization.
      2. Dense Matrix Scan: Leverages C++ FAISS IndexFlatIP to search coordinate dimensions via
         SIMD hardware parallel operations.
      3. Relational Enrichment: Performs single-hop bulk extraction from PostgreSQL via `ANY (%s)` arrays.
    """
    try:
        _init_multilingual_cache()

        if not query.strip():
            return []

        # Generate structural continuous embedding vector
        # normalize_embeddings=True enforces vector bounds magnitude to 1.0, enabling Cosine similarity mechanics
        query_vector = _multi_model.encode([query], normalize_embeddings=True)
        query_vector_faiss = np.array(query_vector).astype('float32')

        # Execute ultra-fast Inner Product nearest neighbor scan
        scores, indices = _multi_faiss_index.search(query_vector_faiss, top_k)

        valid_hits = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:
                doc_id = _multi_doc_ids[idx]
                valid_hits.append((doc_id, score))

        if not valid_hits:
            return []

        # 3. Relational context alignment via bulk fetch block
        conn = psycopg2.connect(host="localhost", port=5432, database="ir_search", user="postgres", password="password")
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

        # Construct an immediate O(1) hash map buffer to structure sorting alignment rules
        text_map = {row[0]: row[1] for row in db_rows}

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
        return [{"doc_id": "Error", "score": 0.0, "snippet": f"Multilingual Core Crash: {str(e)}"}]


def rerank_with_multilingual(query: str, candidate_ids: list, text_mapping: dict) -> list[dict]:
    """
    In-Memory Neural Re-ranking Pipeline utilizing Cross-Lingual Semantic Space weights.

    🎯 IR PARADIGM: RE-RANKING VIA ON-THE-FLY MATRIX MULTIPLICATION
      Instead of scanning a static index tree structure, this function recalculates dynamic
      topological proximity weights between a query string and a pre-filtered candidancy subset
      strictly within the local CPU RAM memory stack.

    📐 MATHEMATICAL FORMATION: VECTOR DOT PRODUCT SIMILARITY
      When dense continuous representations are scaled to unit length (L2 Normalized), the
      geometric Cosine Similarity collapses directly into a linear Dot Product operation.
      Formula:
         Score_ReRanked(q, d) = q • d = sum( q_i * d_i )

      Computational Execution:
         - Let 'query_vec' be a matrix row vector of shape (1, 384).
         - Let 'doc_vecs' be a dense matrix of candidate representations of shape (K, 384).
         - We evaluate similarity instantly using optimized NumPy matrix transposition and multiplication:
           Matrix_Product = doc_vecs × query_vec^T  == Shape: (K, 1)
    """
    try:
        _init_multilingual_cache()
        if not query.strip() or not candidate_ids:
            return []

        # Isolate baseline candidate textual content payloads from incoming tracking dictionaries
        valid_ids, docs = [], []
        for doc_id in candidate_ids:
            text = text_mapping.get(doc_id, "")
            if text:
                valid_ids.append(doc_id)
                docs.append(text)

        if not docs:
            return []

        # Dynamically project query and filtered candidate matrices into the continuous shared space
        query_vec = _multi_model.encode([query], normalize_embeddings=True).astype('float32')
        doc_vecs  = _multi_model.encode(docs,    normalize_embeddings=True).astype('float32')

        # Enforce high-speed Vector Dot Product via transposition multiplication mapping (.T)
        scores    = np.dot(doc_vecs, query_vec.T).flatten()

        # Synthesis results mapping layout
        results = [
            {"doc_id": doc_id, "score": float(score), "snippet": text_mapping[doc_id]}
            for doc_id, score in zip(valid_ids, scores)
        ]

        # Enforce structural descending priority rank adjustments
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    except Exception as e:
        return [{"doc_id": "Error", "score": 0.0, "snippet": f"Multilingual Rerank Crash: {str(e)}"}]