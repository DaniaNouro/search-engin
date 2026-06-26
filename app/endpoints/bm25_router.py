from fastapi import APIRouter, Query, HTTPException
from services.retrieval.bm25_retriever import search_bm25

router = APIRouter(
    prefix="/search",
    tags=["Lexical Search Engines"]
)

@router.post("/bm25")
def api_search_bm25(
        query: str = Query(..., description="The textual query entered by the user"),
        top_k: int = Query(10, description="Number of top matching documents to retrieve"),

        k1: float = Query(1.5, description="BM25 term frequency saturation adjustment"),
        b: float = Query(0.75, description="BM25 document length normalization adjustment")
):
    """
    Executes an online runtime search using the Probabilistic Okapi BM25 Model with active hyperparameter tuning.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    results = search_bm25(query=query, top_k=top_k, k1=k1, b=b)

    if results and results[0]["doc_id"] == "Error":
        raise HTTPException(status_code=500, detail=results[0]["snippet"])

    return results