from fastapi import APIRouter, Query, HTTPException
from services.retrieval.bert_retriever import search_bert

router = APIRouter(
    prefix="/search",
    tags=["Semantic Search Engines"]
)

@router.post("/bert")
def api_search_bert(
    query: str = Query(..., description="The textual query entered by the user"),
    top_k: int = Query(10, description="Number of top matching semantic documents to retrieve")
):
    """
    Executes an online runtime semantic search using Sentence-BERT and FAISS vector indexing.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")


    results = search_bert(query=query, top_k=top_k)

    if results and results[0]["doc_id"] == "Error":
        raise HTTPException(status_code=500, detail=results[0]["snippet"])

    return results