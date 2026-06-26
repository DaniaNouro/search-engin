"""
Module Name: tfidf_router.py
Purpose: FastAPI router defining HTTP endpoints for the TF-IDF retrieval service.
"""

from fastapi import APIRouter, Query, HTTPException
from services.retrieval.tfidf_retriever import search_tfidf

router = APIRouter(
    prefix="/search",
    tags=["Lexical Search Engines"]
)


@router.post("/tfidf")
def api_search_tfidf(
        query: str = Query(..., description="The textual query entered by the user"),
        top_k: int = Query(10, description="Number of top matching documents to retrieve")
):
    """
    Executes an online runtime search using the Lexical Vector Space Model (TF-IDF).
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    results = search_tfidf(query=query, top_k=top_k)

  
    if results and results[0]["doc_id"] == "Error":
        raise HTTPException(status_code=500, detail=results[0]["snippet"])

    return results