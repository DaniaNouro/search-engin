# High-Performance Hybrid & Semantic Information Retrieval Engine

multi-strategy Information Retrieval (IR) system optimized for large-scale textual corpora (522K+ documents).
This architecture orchestrates traditional probabilistic sparse retrieval with modern deep learning dense retrieval, supporting cross-lingual alignment and advanced multi-stage re-ranking.

---

##  System Architecture & Core Pipelines

The system is split into two distinct execution lifecycles: **Offline Heavy Computation (ETL & Indexing)** and **Online Runtime Low-Latency Retrieval**.



### 1. The Offline Ingestion & Indexing Pipeline
* **Extraction & Light Cleaning (`store_clean.py`):** Streams raw corpora from PostgreSQL in strict linear batches ($O(\text{Batch Size})$) via a dual-cursor model to eliminate memory spikes. Text undergoes case folding, punctuation stripping, and whitespace collapse.
* **Vector Embeddings Projection (`train_bert_index.py` & `build_multilingual_index.py`):** Transformer Bi-encoders project processed textual contexts into continuous latent semantic spaces ($\mathbb{R}^{384}$).
* **ANN Matrix Indexing (FAISS Binary Serialization):** Embeddings are subjected to $L_2$ normalization, forcing them onto a unit hypersphere surface. They are then injected into a C++ optimized `IndexFlatIP` (Inner Product) FAISS instance, making the search operation converge exactly to **Cosine Similarity** at hardware register speeds.

### 2. The Online Search Execution Engine
* **Singletons In-Memory Caching:** Serialized structures, index coordinate trees, and vocabulary weights are loaded as application-level Singletons during boot to bypass multi-threaded disk I/O bottlenecks.
* **Topic Space Pruning (Whitelisting Over-sampling):** Resolves relational filtering inside dense indexes by over-sampling candidate pools ($K \times 20$) and executing high-speed $O(1)$ set intersection checks in the RAM heap.

---

## 🗂️ Project Directory Structure

```text
ir-search-engine/
│
├── data/
│   ├── vector_store/               # Serialized FAISS binaries (.bin) and ID mappings (.joblib)
│   └── evaluation_results/         # Auto-persisted evaluation metrics JSON payloads
│
├── services/
│   ├── preprocessing/
│   │   └── base_cleaner.py         # Regular expression-driven text cleansing pipelines
│   │
│   ├── retrieval/
│   │   ├── tfidf_retriever.py      # Statistical Vector Space Model baseline
│   │   ├── bm25_retriever.py       # Probabilistic Best Matching 25 engine
│   │   ├── bert_retriever.py       # Monolingual Dense Semantic Retrieval with Topic Pruning
│   │   ├── multilingual_retriever.py# Cross-lingual shared semantic space retriever
│   │   └── hybrid_retriever.py     # Parallel Score Fusion & Serial Cascading Re-ranker
│   │
│   └── evaluation/
│       └── evaluation_service.py   # Centralized IR validation suite (MAP, nDCG, Latency)
│
├── scripts/
│   ├── store_clean.py              # Streaming ETL database script
│   ├── train_bert_index.py         # Offline BERT indexing job
│   └── build_multilingual_index.py # Offline Multilingual alignment indexing job
│
├── .gitignore                      # Strict resource guarding configuration
├── requirements.txt                # Fixed framework dependency manifests
└── README.md                       # System architecture documentation
🎯 Retrieval Models Stack (The Hierarchy)The search layer implements four dimensional paradigms to serve distinct search characteristics:

                  ┌────────────────────────────────────────┐
                  │          Hybrid Retrieval              │
                  │   (Parallel Fusion / Serial Cascade)   │
                  └───────────────────┬────────────────────┘
                                      │
             ┌────────────────────────┴────────────────────────┐
             ▼                                                 ▼
┌────────────────────────┐                        ┌────────────────────────┐
│     Dense Neural       │                        │     Sparse Lexical     │
│   (BERT Embeddings)    │                        │  (BM25 Probabilistic)  │
└────────────┬───────────┘                        └────────────┬───────────┘
             │                                                 │
             ▼                                                 ▼
┌────────────────────────┐                        ┌────────────────────────┐
│   Multilingual Space   │                        │    TF-IDF baseline     │
│  (Cross-Lingual MiniLM)│                        │  (Term Frequency VSM)  │
└────────────────────────┘                        └────────────────────────┘

🛠️ Installation & Execution Guidelines1.

Environment SetupBash# Instantiate local virtual frame
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install deterministic package dependencies
pip install -r requirements.txt
2. Run Offline Pipelines (Chronological Order)Bash# Step 1: Run streaming text preprocessing and loading
python scripts/store_clean.py

# Step 2: Generate monolingual BERT vectors and FAISS database index
python scripts/train_bert_index.py

# Step 3: Build the global cross-lingual shared index
python scripts/build_multilingual_index.py
3. Run Benchmark ValidationsBash# Execute exhaustive matrix testing over a target model (e.g., Hybrid Parallel)


python services/evaluation/evaluation_service.py HYBRID_PARALLEL
