"""
Module Name: train_tfidf.py
Purpose: Offline training pipeline for the TF-IDF Vector Space Model.
         Streams preprocessed documents from PostgreSQL, builds the sparse matrix, 
         and serializes components using joblib into the centralized data store.
"""

import os
import joblib
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer

# Import the domain-adaptive specialized text normalization pipeline
from services.preprocessing.tfidf_cleaner import tfidf_custom_tokenizer

# Centralized data store absolute path orchestration
OUTPUT_DIR = os.path.join("data", "vector_store")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VECTORIZER_PATH = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(OUTPUT_DIR, "tfidf_matrix.joblib")
DOC_IDS_PATH = os.path.join(OUTPUT_DIR, "doc_ids.joblib")


def build_tfidf_index():
    """
    [OFFLINE JOB]
    Extracts normalized document collections from the relational engine, builds
    the spatial Term-Document matrix using non-linear statistical weights, and
    serializes indexing primitives to disk for real-time downstream retrieval.

    Linguistic & Structural Pipeline:
      1. Extraction: Queries clean text representations from PostgreSQL to bound operational RAM.
      2. Alignment: Maps document identifiers to textual payloads in parallel arrays.
      3. Matrix Synthesis: Fits the VSM vectorizer, passing terms through our customized
         Tokenization/Stemming engine while forcing `lowercase=False` to prevent redundant computations.
      4. Serialization: Persists the vectorizer metadata, compressed sparse matrices, and ID indices.
    """
    print("⏳ Connecting to PostgreSQL to fetch cleaned documents for TF-IDF...")

    try:
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
        print(f"❌ Database connection failed: {e}")
        return

    if not rows:
        print("❌ No processed documents found. Please run store_clean.py pipeline first!")
        return

    # Deconstruct records into aligned parallel data arrays
    doc_ids = [row[0] for row in rows]
    corpus = [row[1] if row in rows and row[1] is not None else "" for row in rows]

    print(f"📊 Successfully extracted {len(corpus):,} documents.")
    print("🧠 Training TfidfVectorizer and building sparse matrix (This may take a moment)...")

    # ─────────────────────────────────────────────────────────────────────────
    # 📐 MATHEMATICAL FOUNDATION & COMPUTATIONAL IR CONCEPTS
    # ─────────────────────────────────────────────────────────────────────────
    # This phase constructs a high-dimensional Vector Space Model (VSM) mapping
    # documents into spatial vectors based on statistical term importance.
    #
    # 1. Term Frequency (TF): Measures local text importance inside a document.
    #    Formula: TF(t, d) = frequency count of term 't' in document 'd'.
    #
    # 2. Inverse Document Frequency (IDF): Penalizes generic background vocabulary.
    #    Formula: IDF(t) = log( (1 + N) / (1 + DF(t)) ) + 1
    #    Where 'N' is the total collection size and 'DF(t)' is the document frequency.
    #
    # 3. Comprehensive Term Weighting:
    #    Formula: TF-IDF(t, d, D) = TF(t, d) * IDF(t)
    #
    # 4. Spatial Normalization (Euclidean / L2 Norm):
    #    Ensures long and short documents are projected equivalently into the vector room.
    #    Formula: V_norm = V / sqrt( sum(v_i ^ 2) )
    #
    # 5. Compressed Sparse Row (CSR) Optimization:
    #    Since most terms in the vocabulary do not appear in every document, the matrix
    #    is highly sparse (mostly zeros). TfidfVectorizer outputs a scipy CSR matrix
    #    which dramatically reduces memory footprint by storing only non-zero coordinates.
    # ─────────────────────────────────────────────────────────────────────────

    # Instantiate the vector space model with our customized pipeline tokenizer
    vectorizer = TfidfVectorizer(tokenizer=tfidf_custom_tokenizer, lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    print("💾 Serializing and saving index files via joblib...")

    # Safely persist binary representations for sub-millisecond production inference
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(tfidf_matrix, MATRIX_PATH)
    joblib.dump(doc_ids, DOC_IDS_PATH)

    print("🏁 [SUCCESS] TF-IDF Offline Indexing completed successfully!")
    print(f" ├── Vectorizer saved to: {VECTORIZER_PATH}")
    print(f" ├── TF-IDF Matrix saved to: {MATRIX_PATH}")
    print(f" └── Document IDs saved to: {DOC_IDS_PATH}")


if __name__ == "__main__":
    print("=== Starting TF-IDF Offline Indexing Job ===")
    build_tfidf_index()