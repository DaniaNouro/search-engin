"""
Module Name: build_multilingual_index.py
Purpose: Offline indexing pipeline for the Extra Multilingual Feature.
         Fetches processed documents from PostgreSQL, generates multilingual embeddings,
         and saves FAISS index + IDs mapping to disk.
"""

import os
import sys
import joblib
import psycopg2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Orchestrate project-level root directories for systemic visibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STORE_DIR = os.path.join("data", "vector_store")
MULTI_FAISS_INDEX_PATH = os.path.join(STORE_DIR, "multilingual_faiss_index.bin")
MULTI_IDS_PATH = os.path.join(STORE_DIR, "multilingual_doc_ids.joblib")


def build_index():
    """
    [OFFLINE MULTILINGUAL JOB]
    Extracts structural textual entries, maps their global semantic weights
    using a multi-language aligned Transformer topology, maps cross-lingual tokens,
    and flushes a unified high-speed binary FAISS matrix space to disk.
    """
    # 1. Enforce absolute output directory structure validation
    os.makedirs(STORE_DIR, exist_ok=True)

    print("⏳ Connecting to PostgreSQL database...")
    try:
        conn = psycopg2.connect(
            host="localhost", port=5432, database="ir_search", user="postgres", password="password"
        )
        cursor = conn.cursor()

        # Extract relational primary keys and clean context tokens in a single batch query
        print("📥 Fetching documents from 'processed_documents'...")
        cursor.execute(
            "SELECT doc_id, clean_text FROM processed_documents WHERE clean_text IS NOT NULL AND clean_text != '';")
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        if not rows:
            print("❌ Error: No processed documents found in the database. Run preprocessing first.")
            return

        doc_ids = [row[0] for row in rows]
        documents = [row[1] for row in rows]
        print(f"✅ Successfully fetched {len(documents)} documents.")

        # 2. Instantiate the Cross-Lingual Knowledge Distillation Model
        # IR Paradigm: Multilingual Retrieval — bridging language boundaries without external translation steps.
        print("🧠 Loading Multilingual SentenceTransformer Model ('paraphrase-multilingual-MiniLM-L12-v2')...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # ─────────────────────────────────────────────────────────────────
        # 📐 MATHEMATICAL FOUNDATION: CROSS-LINGUAL ALIGNED VECTOR SPACE
        # ─────────────────────────────────────────────────────────────────
        # Traditional monolingual models separate linguistic spaces entirely.
        # This implementation leverages Cross-lingual Knowledge Distillation to map
        # distinct natural languages into a single, shared continuous vector room (R^384).
        #
        # 1. Semantic Invariance:
        #    The model maps parallel concepts across 50+ languages to identical coordinates.
        #    Example: The semantic vectors for "Search Engine", "محرك بحث", and "Moteur de recherche"
        #    converge to the same spatial neighborhood under this mapping projection.
        #
        # 2. Vector L2 Normalization (Geometric Cosine Convergence):
        #    Setting `normalize_embeddings=True` standardizes vector scaling properties:
        #    Formula: V_normalized = V / ||V||
        #    This forces all coordinate vectors onto a shared unit hypersphere boundary.
        #
        # 3. Flat Inner Product (IP) Index Matching:
        #    For L2-normalized spaces, the Inner Product (Dot Product) exactly corresponds
        #    to the Cosine Similarity calculation, optimizing computational resources.
        #    Formula: Distance(A, B) = sum( a_i * b_i ) = Cosine_Similarity(A, B)
        # ─────────────────────────────────────────────────────────────────
        print("⚡ Generating dense vector embeddings (this might take a few moments)...")
        embeddings = model.encode(documents, show_progress_bar=True, normalize_embeddings=True)
        embeddings_faiss = np.array(embeddings).astype('float32')

        # 4. Construct the High-Performance C++ FAISS Vector Space Instance
        vector_dimension = embeddings_faiss.shape[1]
        print(f"🧱 Initializing FAISS IndexFlatIP with dimension: {vector_dimension}...")
        index = faiss.IndexFlatIP(vector_dimension)

        # Append computed semantic continuous blocks into the binary index space
        print("➕ Adding embeddings to FAISS Index...")
        index.add(embeddings_faiss)

        # 5. Serialize physical memory matrices into centralized disk storage blocks
        print(f"💾 Saving FAISS binary index to: {MULTI_FAISS_INDEX_PATH}")
        faiss.write_index(index, MULTI_FAISS_INDEX_PATH)

        print(f"💾 Saving document IDs mapping to: {MULTI_IDS_PATH}")
        joblib.dump(doc_ids, MULTI_IDS_PATH)

        print("🎉 [SUCCESS] Multilingual offline indexing completed successfully!")
        print(f"📊 Total Indexed Documents: {index.ntotal}")

    except Exception as e:
        print(f"❌ Critical Error during offline indexing: {str(e)}")


if __name__ == "__main__":
    build_index()