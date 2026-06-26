#import os
from fastapi import APIRouter
from services.retrieval.hybrid_retriever import search_serial_hybrid, search_parallel_hybrid

router = APIRouter(
    prefix="/search",
    tags=["Search Engine Endpoints"]
)


@router.post("/hybrid")
def api_search_hybrid(
        query: str,
        top_k: int = 10,
        hybrid_mode: str = "parallel",
        alpha: float = 0.5,
        beta: float = 0.5,
        multilingual: bool = False
):
    """
    [HYBRID SEARCH ENDPOINT]
    Routes the query directly to either Parallel or Serial hybrid retrieval systems.
    """
    try:

        if hybrid_mode.strip().lower() == "serial":
            results = search_serial_hybrid(
                query=query,
                top_k=top_k,
                multilingual=multilingual
            )
        else:
            results = search_parallel_hybrid(
                query=query,
                top_k=top_k,
                alpha=alpha,
                beta=beta,
                multilingual=multilingual
            )

        return results

    except Exception as e:
        return [{"doc_id": "Error", "score": 0.0, "snippet": f"Hybrid Search Error: {str(e)}"}]