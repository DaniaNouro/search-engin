"""
Module Name: evaluation_service.py
Purpose: Centralized validation framework to evaluate structural textual information
         retrieval architectures (MAP, Precision@K, Recall@K, nDCG) globally over the full
         qrels corpus footprint without random subset statistical sampling noise.
"""

import time
import math
import json
import os
import numpy as np
import psycopg2

from services.retrieval.tfidf_retriever import search_tfidf
from services.retrieval.bm25_retriever import search_bm25
from services.retrieval.bert_retriever import search_bert, _init_bert_cache
from services.retrieval.hybrid_retriever import search_parallel_hybrid, search_serial_hybrid


def compute_ndcg(hits: list, k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank K (nDCG@K).

    IR Mathematical Principle:
      Measures graded relevance utility, introducing a logarithmic decay penalty
      proportional to document positional rank depth. Documents matching ground truth
      at top ranks contribute higher utility scores than lower ranked items.
    """
    # DCG evaluation loop applying strict logarithmic scaling positional dividers
    dcg   = sum(h / math.log2(r + 1) for r, h in enumerate(hits[:k], 1))

    # Computing Ideal DCG (IDCG) boundary limits assuming perfect descending sorting order
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(k, sum(hits)) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def load_ground_truth() -> dict:
    """
    [DATA ENGINE LOGIC]
    Extracts all benchmarking query criteria entries and their positive binary
    relevance intersection pairs from the relational PostgreSQL state without data pooling limits.
    """
    try:
        conn = psycopg2.connect(
            host="localhost", port=5432, database="ir_search",
            user="postgres", password="password"
        )
        cursor = conn.cursor()

        # Enforces zero sampling limitations to extract the exhaustive evaluation horizon
        cursor.execute("""
            SELECT DISTINCT q.query_id, q.query_text
            FROM queries q
            JOIN qrels r ON q.query_id = r.query_id
            ORDER BY q.query_id;
        """)
        queries = cursor.fetchall()
        total = len(queries)
        print(f"📊 Total queries loaded from qrels: {total}")

        ground_truth = {}
        for query_id, query_text in queries:
            cursor.execute("""
                SELECT doc_id FROM qrels 
                WHERE query_id = %s AND relevance > 0;
            """, (query_id,))
            rel_docs = [row[0] for row in cursor.fetchall()]
            if rel_docs:
                # String conversion layer unifies downstream matrix comparisons bounds
                ground_truth[query_id] = {
                    "text": query_text,
                    "relevant_docs": set(str(d) for d in rel_docs)
                }

        cursor.close()
        conn.close()
        return ground_truth

    except Exception as e:
        print(f"❌ Error loading ground truth: {e}")
        return {}


def evaluate_model(model_name: str = "TF-IDF", top_k: int = 10, **kwargs) -> dict:
    """
    Iterates exhaustively through the ground truth framework to score information accuracy
    profiles (MAP, P@K, R@K, nDCG) and runtime latency benchmarks under identical contexts.
    """
    current_model = str(model_name).strip().upper()
    ground_truth  = load_ground_truth()

    if not ground_truth:
        return {"Error": "No queries loaded or DB connection failed."}

    # ── Cold Start Isolation Countermeasure ──
    # Instantiates deep transformers model layers prior to tracking execution loop intervals,
    # preventing disk serialization latency bounds from inflating pure online query metrics.
    if current_model in ["BERT", "HYBRID_PARALLEL", "HYBRID_SERIAL"]:
        try:
            _init_bert_cache()
        except Exception as e:
            return {"Error": f"Failed to initialize BERT: {str(e)}"}

    precisions, recalls, aps, ndcgs, latencies = [], [], [], [], []
    total_queries = len(ground_truth)

    print(f"🚀 Starting evaluation: model='{current_model}' | queries={total_queries} | top_k={top_k}")

    for i, (query_id, data) in enumerate(ground_truth.items(), 1):
        query_text      = data["text"]
        actual_relevant = data["relevant_docs"]

        # ── Latency Measurement Interceptor ──
        start = time.time()

        if current_model == "TF-IDF":
            results = search_tfidf(query_text, top_k=top_k)
        elif current_model == "BM25":
            results = search_bm25(query_text, top_k=top_k)
        elif current_model == "BERT":
            results = search_bert(query_text, top_k=top_k)
        elif current_model == "HYBRID_PARALLEL":
            results = search_parallel_hybrid(query_text, top_k=top_k)
        elif current_model == "HYBRID_SERIAL":
            results = search_serial_hybrid(query_text, top_k=top_k)
        else:
            results = []

        # Convert temporal tracking differentials directly to standard milliseconds (ms)
        latencies.append((time.time() - start) * 1000)

        # Map predictions to tracking lists while filtering system error flags
        predicted = [
            str(r["doc_id"]) for r in results
            if r.get("doc_id") != "Error"
        ]
        hits = [1 if d in actual_relevant else 0 for d in predicted]

        # Precision@K Logic Implementation
        precisions.append(sum(hits) / top_k if top_k > 0 else 0)

        # Recall@K Logic Implementation
        recalls.append(sum(hits) / len(actual_relevant) if actual_relevant else 0)

        # Average Precision (AP) Accumulation Logic
        running, ap_vals = 0, []
        for rank, hit in enumerate(hits, 1):
            if hit:
                running += 1
                ap_vals.append(running / rank)
        aps.append(np.mean(ap_vals) if ap_vals else 0.0)

        # nDCG@K Assignment Processing
        ndcgs.append(compute_ndcg(hits, top_k))

        if i % 500 == 0:
            print(f"   ├── Progress: {i}/{total_queries} queries done...")

    # Consolidate analytical statistical averages across total evaluation frames
    metrics = {
        "model":               current_model,
        "total_queries_used":  total_queries,
        "top_k":               top_k,
        "MAP":                 round(float(np.mean(aps)),        4),
        f"Precision@{top_k}":  round(float(np.mean(precisions)), 4),
        f"Recall@{top_k}":     round(float(np.mean(recalls)),    4),
        "nDCG":                round(float(np.mean(ndcgs)),      4),
        "Avg_Latency_ms":      round(float(np.mean(latencies)),  4),
    }

    # Automatically persist results schema blocks to disk to secure quick presentation delivery
    _save_results(metrics)

    print(f"\n✅ Evaluation complete for {current_model}:")
    print(f"   total_queries_used : {total_queries}")
    print(f"   MAP                : {metrics['MAP']}")
    print(f"   nDCG               : {metrics['nDCG']}")
    print(f"   Precision@{top_k}   : {metrics[f'Precision@{top_k}']}")
    print(f"   Recall@{top_k}      : {metrics[f'Recall@{top_k}']}")
    print(f"   Avg Latency (ms)   : {metrics['Avg_Latency_ms']}")

    return metrics


def _save_results(metrics: dict):
    """Saves final benchmarking metrics structures onto disk stores to secure offline analysis."""
    save_dir = os.path.join("data", "evaluation_results")
    os.makedirs(save_dir, exist_ok=True)

    model_name = metrics.get("model", "unknown").lower()
    path = os.path.join(save_dir, f"eval_{model_name}.json")

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"💾 Results saved to: {path}")


def load_saved_results(model_name: str) -> dict:
    """Retrieves pre-computed evaluation state profiles from local store structures."""
    path = os.path.join("data", "evaluation_results", f"eval_{model_name.lower()}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "BM25"
    print(f"=== Running Evaluation: {model} ===")
    evaluate_model(model_name=model, top_k=10)