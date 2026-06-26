"""
Module Name: train_bert_index.py
Purpose: Offline heavy computation script to generate BERT embeddings
         for 522K documents and build a high-speed FAISS vector index.
"""

import os
import sys
import time
import joblib
import numpy as np
import psycopg2
import faiss
from sentence_transformers import SentenceTransformer

# Orchestrate project-level root directories for systemic visibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enforce strict output paths within the unified data lakehouse architecture
STORE_DIR = os.path.join("data", "vector_store")
os.makedirs(STORE_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(STORE_DIR, "bert_faiss_index.bin")
BERT_IDS_PATH = os.path.join(STORE_DIR, "bert_doc_ids.joblib")

# Chunk capacity bounding parameters to match GPU/CPU core capabilities safely
BATCH_SIZE = 5000


def train_bert_pipeline():
    """
    [OFFLINE HEAVY JOB]
    Streams cleaned textual corpora from PostgreSQL using server-side resource caching,
    projects contextual semantics into a 384-dimensional dense room using a Bi-Encoder Transformer,
    and indexes latent topological spaces inside an optimized FAISS vector engine.
    """
    print("🚀 Starting Offline BERT + FAISS Indexing Pipeline...")

    # 1. Instantiate the Deep Transformer Embedding Engine
    # IR Paradigm: Neural Retrieval — mapping semantic contexts beyond surface vocabulary matching.
    print("⏳ Loading BERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = 384  # Dimensionality factor of the continuous semantic vector room
    print(f"✅ Model loaded. Embedding dimension is: {embedding_dim}")

    # 2. Build the FAISS Vector Index Structure
    # IR Concept: Approximate Nearest Neighbor (ANN) Indexing via IndexFlatIP (Inner Product).
    # When combined with L2-normalized embeddings, Inner Product exactly measures Cosine Similarity.
    faiss_index = faiss.IndexFlatIP(embedding_dim)

    # Parallel tracking array to align matrix coordinate indices with relational primary keys
    all_doc_ids = []

    # 3. Connection orchestration using streaming relational blocks (Server-Side Cursor)
    try:
        conn = psycopg2.connect(
            host="localhost", port=5432, database="ir_search", user="postgres", password="password"
        )
        # Using a named server-side cursor prevents loading all 522K documents into raw RAM at once,
        # ensuring linear memory consumption behavior O(Batch_Size).
        cursor = conn.cursor(name="bert_stream_cursor")
        cursor.execute(
            "SELECT doc_id, clean_text FROM processed_documents WHERE clean_text IS NOT NULL AND clean_text != '';")

        print("🔗 Database connection established. Processing streaming batches...")

        batch_texts = []
        batch_ids = []
        total_processed = 0
        start_time = time.time()

        while True:
            rows = cursor.fetchmany(BATCH_SIZE)
            if not rows:
                break  # Corpus streaming complete

            for row in rows:
                batch_ids.append(row[0])
                batch_texts.append(row[1])

            # ─────────────────────────────────────────────────────────────────
            # 📐 MATHEMATICAL FOUNDATION: DENSE EMBEDDINGS & CONTINUOUS VSM
            # ─────────────────────────────────────────────────────────────────
            # Unlike sparse representations (TF-IDF/BM25) which suffer from vocabulary mismatch
            # and curse of dimensionality, BERT models project continuous textual context into
            # a compact, dense mathematical vector room: R^384.
            #
            # 1. Deep Semantic Encoding:
            #    Textual strings are mapped via self-attention heads to isolate latent concepts,
            #    resolving synonymy ("car" mapping to "automobile") and polysemy.
            #
            # 2. Vector L2 Normalization (The Cosine Equivalence):
            #    Setting `normalize_embeddings=True` structurally divides each continuous vector
            #    by its Euclidean norm: V_norm = V / ||V||
            #    Effect: Forcing all document vector lengths to exactly 1.0 positions them onto
            #    the surface of a unit hypersphere.
            #
            # 3. Inner Product Index Matching (FAISS IndexFlatIP):
            #    For L2-normalized vectors, the Inner Product (Dot Product) mathematically converges
            #    to match the Cosine Similarity exactly:
            #    Formula: Dot_Product(A, B) = sum( a_i * b_i ) = Cosine_Similarity(A, B)
            #
            # 4. Hardware Optimization:
            #    The underlying execution matrix operations are cast strictly to `float32`,
            #    allowing SIMD/AVX hardware registers or GPU CUDA alignments to score vector boundaries
            #    substantially faster than standard relational architectures.
            # ─────────────────────────────────────────────────────────────────
            embeddings = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)

            # Cast arrays to float32 explicitly to comply with low-level FAISS binary alignments
            embeddings_faiss = np.array(embeddings).astype('float32')

            # Inject dense embeddings into the continuous geometric index engine
            faiss_index.add(embeddings_faiss)

            # Store matching tracking index sequences
            all_doc_ids.extend(batch_ids)

            total_processed += len(batch_ids)
            print(
                f"📊 Processed: {total_processed} documents... (Speed: {total_processed / (time.time() - start_time):.2f} docs/sec)")

            # Clear temporal heap frames immediately to release memory allocations to the OS GC
            batch_texts = []
            batch_ids = []

        cursor.close()
        conn.close()

        # 4. Binary serialization and data store flushing
        print("\n💾 Saving FAISS Vector Index and Document IDs mapping...")
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        joblib.dump(all_doc_ids, BERT_IDS_PATH)

        print(f"🎉 Pipeline successfully completed!")
        print(f"📁 Vector Index Saved To: {FAISS_INDEX_PATH}")
        print(f"📁 Document Mapping Saved To: {BERT_IDS_PATH}")
        print(f"⏱️ Total Execution Time: {(time.time() - start_time) / 60:.2f} minutes.")

    except Exception as e:
        print(f"❌ Critical Pipeline Failure: {str(e)}")


if __name__ == "__main__":
    train_bert_pipeline()