"""
Module Name: train_bm25.py
Purpose: Offline indexing pipeline for Okapi BM25 Probabilistic Model.
         Extracts cleaned text from PostgreSQL, tokenizes the corpus,
         trains the BM25Okapi model, and serializes assets via joblib.
"""

import os
import sys
import joblib
import psycopg2
from rank_bm25 import BM25Okapi

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.preprocessing.bm25_cleaner import bm25_custom_tokenizer

# Enforce explicit output directory mapping inside the unified data lakehouse
STORE_DIR = os.path.join("data", "vector_store")
BM25_MODEL_PATH = os.path.join(STORE_DIR, "bm25_model.joblib")
BM25_IDS_PATH = os.path.join(STORE_DIR, "bm25_doc_ids.joblib")


def run_bm25_offline_indexing():
    """
    [OFFLINE JOB]
    Extracts structural textual representations, builds inverted index stats,
    computes algorithmic constants (collection length limits), and flushes
    the binary state to disk for sub-millisecond retrieval execution.
    """
    print("=== Starting BM25 Offline Indexing Job ===")
    os.makedirs(STORE_DIR, exist_ok=True)

    # 1. Connect to PostgreSQL to pull baseline documents
    try:
        print("⏳ Connecting to PostgreSQL to fetch cleaned documents for BM25...")
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="ir_search",
            user="postgres",
            password="password"
        )
        cursor = conn.cursor()

        print("📥 Executing query on 'processed_documents' table...")
        cursor.execute("SELECT doc_id, clean_text FROM processed_documents;")
        rows = cursor.fetchall()

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ [CRITICAL ERROR] Database connection failed: {str(e)}")
        return

    doc_ids = []
    tokenized_corpus = []

    print(f"📊 Successfully extracted {len(rows)} documents. Starting Tokenization...")

    # 2. Build the Tokenized Corpus (Translating structural strings into sequence token arrays)
    for count, row in enumerate(rows, 1):
        doc_id, clean_text = row
        doc_ids.append(doc_id)

        # Tokenization executed here via the symmetry wrapper
        tokens = bm25_custom_tokenizer(clean_text if clean_text else "")
        tokenized_corpus.append(tokens)

        if count % 100000 == 0:
            print(f"   ├── Processed {count} / {len(rows)} documents...")

    # ─────────────────────────────────────────────────────────────────────────
    # 📐 MATHEMATICAL FOUNDATION: THE OKAPI BM25 PROBABILISTIC MODEL
    # ─────────────────────────────────────────────────────────────────────────
    # BM25 is a non-linear state-of-the-art probabilistic retrieval function that
    # optimizes term weighting based on local saturation and collection document length constraints.
    # Unlike basic TF-IDF, BM25 prevents an extreme term frequency from dominating the document weight.
    #
    # Core Mathematical Components computed during this fitting block:
    #
    # 1. Robertson-Spärck Jones IDF (Inverse Document Frequency variant):
    #    Formula: IDF(q_i) = ln( (N - DF(q_i) + 0.5) / (DF(q_i) + 0.5) + 1 )
    #    Where 'N' is the total collection count, and 'DF(q_i)' is the document frequency of term 'q_i'.
    #    - If a term appears in more than half the database documents, this IDF can approach 0.
    #
    # 2. Non-linear Term Frequency Saturation (The Local Document TF Weight):
    #    Formula Component: ( TF(q_i, d) * (k1 + 1) ) / ( TF(q_i, d) + k1 * (1 - b + b * (dl / avgdl)) )
    #
    #    Parameters Explained:
    #    - TF(q_i, d): Local raw count of term 'q_i' inside document 'd'.
    #    - dl: The explicit length of the current document 'd' (token count).
    #    - avgdl: The global Average Document Length computed across the entire corpus collection.
    #
    #    Hyperparameters (Tunable at runtime/training via UI configuration):
    #    - k1 (Term Frequency Saturation limit): Typically in [1.2, 2.0]. Controls the scale bounds
    #      of term frequency growth. As TF increases, the local score approaches an asymptote (saturates).
    #    - b (Document Length Normalization penalty): Typically in [0.5, 0.8]. Penalizes long
    #      documents that contain repetitive noise or accidental multi-term matches.
    #      * b = 1.0 -> Absolute scale-based length penalty applied.
    #      * b = 0.0 -> Complete removal of document length variance consideration.
    # ─────────────────────────────────────────────────────────────────────────
    print("🧠 Training BM25Okapi instance and building frequency maps (This may take a moment)...")
    bm25_model = BM25Okapi(tokenized_corpus)

    # 4. Serialize built objects into disk blocks using optimized compress streams
    print("💾 Serializing and saving BM25 index files via joblib...")
    try:
        # Compression level 3 balances serialization disk space usage and I/O latency
        joblib.dump(bm25_model, BM25_MODEL_PATH, compress=3)
        joblib.dump(doc_ids, BM25_IDS_PATH, compress=3)

        print("🏁 [SUCCESS] BM25 Offline Indexing completed successfully!")
        print(f" ├── BM25 Model saved to: {BM25_MODEL_PATH}")
        print(f" └── Document IDs saved to: {BM25_IDS_PATH}")
    except Exception as e:
        print(f"❌ [ERROR] Serialization failed: {str(e)}")


if __name__ == "__main__":
    run_bm25_offline_indexing()