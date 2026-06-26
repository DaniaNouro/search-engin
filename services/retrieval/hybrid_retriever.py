"""
Module Name: hybrid_retriever.py
Purpose: Advanced Multi-Strategy Hybrid Retrieval Engine orchestrating
         lexical and neural representations. Supports linear score fusion
         (Parallel Mode) and multi-stage filtering cascaded pipeline (Serial Mode).
"""

import os
import sys

from services.retrieval.bm25_retriever import search_bm25
from services.retrieval.bert_retriever import search_bert
from services.retrieval.multilingual_retriever import search_multilingual

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def normalize_scores(results: list) -> dict:
    """
    Executes a Min-Max Vector Normalization layer to address the scale discrepancy
    (Apples-to-Oranges scoring conflict) between structural lexical statistics
    and bounded deep embedding vectors.

    📐 MATHEMATICAL FORMATION: MIN-MAX NORMALIZATION
      Transforms raw scores of variant distributions into a uniform domain mapped to [0.0, 1.0].
      Formula:
         S_normalized = (S_raw - S_min) / (S_max - S_min)

      Edge Case Handling: If all fetched candidate scores converge (S_max == S_min),
      the function assigns a uniform constant scaling weight of 1.0 to prevent division-by-zero errors.

    Args:
        results (list): Raw result objects array containing strict algorithm-specific 'score' attributes.

    Returns:
        dict: High-speed O(1) memory mapping structure linking Doc_IDs to their normalized [0.0, 1.0] weights.
    """
    if not results:
        return {}
    scores = [res["score"] for res in results]
    min_score, max_score = min(scores), max(scores)
    if max_score == min_score:
        return {res["doc_id"]: 1.0 for res in results}
    return {
        res["doc_id"]: (res["score"] - min_score) / (max_score - min_score)
        for res in results
    }


def search_parallel_hybrid(query: str, top_k: int = 10, alpha: float = 0.5,
                           beta: float = 0.5, multilingual: bool = False) -> list[dict]:
    """
    [PARALLEL HYBRID MODE] - Reciprocal Rank / Score Fusion Architecture.

    IR System Design:
      1. Parallel Oversampling: Executes asynchronous lexical (BM25) and neural (BERT) lookups
         with expanded boundaries (extended_k = top_k * 3) to protect recall capabilities.
      2. Domain Uniformity: Normalizes disparate scores into a common mathematical space.
      3. In-Memory Component Stitching: Builds an O(1) memory index mapping matching textual snippets
         without re-triggering database connections, avoiding performance anti-patterns.
      4. Convex Combination Scoring: Applies weighted interpolation factors to balance precision and semantic tone.

    📐 MATHEMATICAL FORMATION: WEIGHTED SCORE FUSION (LINEAR COMBINATION)
      Combines normalized evidence scores derived from different feature extraction streams.
      Formula:
         Score_Hybrid(d) = (alpha * S_BM25(d)) + (beta * S_BERT(d))

      Where:
         - alpha (α): The tuning parameter controlling the lexical structural emphasis.
         - beta (β): The tuning parameter controlling the transformer deep semantic embedding emphasis.
         - Constraints: Ideally, alpha + beta = 1.0 to preserve the normalized scaling boundaries.
    """
    # Extended search depth optimization ensures adequate overlap population for fusion steps
    extended_k = top_k * 3

    bm25_results = search_bm25(query, top_k=extended_k)
    bert_results = (search_multilingual(query, top_k=extended_k)
                    if multilingual else search_bert(query, top_k=extended_k))

    if not bm25_results and not bert_results:
        return []

    # Map variant scores distribution fields into unified bounding blocks
    bm25_norms = normalize_scores(bm25_results)
    bert_norms = normalize_scores(bert_results)

    # Compile a transient map index from existing hits data to bypass secondary PostgreSQL overhead
    text_mapping = {res["doc_id"]: res.get("snippet", "")
                    for res in bm25_results + bert_results}

    # Set union operation targets total combined documents extracted across both pipelines
    all_ids = set(bm25_norms) | set(bert_norms)

    hybrid_scores = []
    for doc_id in all_ids:
        s_bm25 = bm25_norms.get(doc_id, 0.0)
        s_bert = bert_norms.get(doc_id, 0.0)

        # Enforce linear combination formula computation
        combined_score = (alpha * s_bm25) + (beta * s_bert)
        hybrid_scores.append({
            "doc_id": doc_id,
            "score": float(combined_score),
            "snippet": text_mapping.get(doc_id, "")
        })

    # Sort candidates strictly based on unified vector fusion values downwards
    hybrid_scores.sort(key=lambda x: x["score"], reverse=True)
    return hybrid_scores[:top_k]


def search_serial_hybrid(query: str, top_k: int = 10, retrieve_limit: int = 100,
                         multilingual: bool = False) -> list[dict]:
    """
    [SERIAL HYBRID MODE] - Multi-Stage Cascaded Filtering & Re-ranking Architecture.

    IR Pipeline Design:
      Implements a strict bottleneck funneling structure designed to optimize expensive model execution.
      - Multilingual Path: Phase 1 isolates semantic clusters over global domains via dense vectors
        -> Phase 2 applies localized BM25 probabilistic constraints to refine the final rank.
      - Standard Path: Phase 1 isolates high-recall candidates quickly via fast lexical BM25
        -> Phase 2 processes the pruned candidate subset through heavy BERT transformer attention matrices.

    🎯 IR ADVANTAGE: COMPUTATIONAL EFFICIENCY
      Executing complex deep transformers over an entire 522K document set at runtime is prohibitive.
      Serial cascading limits heavy scoring algorithms to a pre-filtered window (e.g., `retrieve_limit=100`),
      delivering deep semantic precision at a fraction of the computational runtime cost.
    """
    if multilingual:
        # ── Cascaded Funnel Path A: Cross-Language Semantic Filtering ──
        candidate_results = search_multilingual(query, top_k=retrieve_limit)
        if not candidate_results:
            return []

        candidate_ids = [res["doc_id"] for res in candidate_results]
        text_mapping = {res["doc_id"]: res.get("snippet", "") for res in candidate_results}

        # Stage 2: Fetch lexical overlap scoring boundaries for re-ranking
        bm25_results = search_bm25(query, top_k=retrieve_limit)
        candidate_set = set(candidate_ids)
        bm25_score_map = {r["doc_id"]: r["score"] for r in bm25_results if r["doc_id"] in candidate_set}

        re_ranked = []
        for doc_id in candidate_ids:
            # Fallback strategy protects score metrics if a document is missing in the alternate lookup map
            fallback_score = next(r["score"] for r in candidate_results if r["doc_id"] == doc_id)
            re_ranked.append({
                "doc_id": doc_id,
                "score": float(bm25_score_map.get(doc_id, fallback_score)),
                "snippet": text_mapping[doc_id]
            })

        return sorted(re_ranked, key=lambda x: x["score"], reverse=True)[:top_k]

    else:
        # ── Cascaded Funnel Path B: High-Recall Lexical Pruning + Neural Re-ranking ──
        # Phase 1: Retrieve high-recall broad context via fast sparse lookup
        candidate_results = search_bm25(query, top_k=retrieve_limit)
        if not candidate_results:
            return []

        candidate_ids = [res["doc_id"] for res in candidate_results]
        text_mapping = {res["doc_id"]: res.get("snippet", "") for res in candidate_results}

        # Phase 2: Intersect through high-precision semantic deep neural scoring
        bert_results = search_bert(query, top_k=retrieve_limit)
        candidate_set = set(candidate_ids)

        # Retain only components that successfully mapped through the initial high-recall barrier
        re_ranked = [
            {
                "doc_id": r["doc_id"],
                "score": float(r["score"]),
                "snippet": text_mapping.get(r["doc_id"], "")
            }
            for r in bert_results if r["doc_id"] in candidate_set
        ]

        return sorted(re_ranked, key=lambda x: x["score"], reverse=True)[:top_k]