"""
Module Name: api_client.py
Purpose: Centralized HTTP client to communication with the FastAPI backend engine.
"""
import requests

BASE_URL = "http://127.0.0.1:8085"


def fetch_search_results(query: str, model: str, top_k: int, k1: float = 1.5, b: float = 0.75,
                         hybrid_mode: str = "parallel", alpha: float = 0.5, beta: float = 0.5,multilingual: bool = False,) -> list:
    """Sends a search request to the backend API with full dynamic hyperparameters."""


    endpoint_map = {
        "TF-IDF": "/search/tfidf",
        "BM25": "/search/bm25",
        "BERT": "/search/bert",
        "Hybrid": "/search/hybrid"
    }

    url = f"{BASE_URL}{endpoint_map.get(model, '/search/tfidf')}"


    params = {
        "query": query,
        "top_k": top_k

    }

    params.update({
        "multilingual": multilingual,
    })

    if model == "BM25":
        params.update({"k1": k1, "b": b})
    elif model == "Hybrid":
        params.update({
            "hybrid_mode": hybrid_mode,
            "alpha": alpha,
            "beta": beta,
        })

    try:

        response = requests.post(url, params=params)
        if response.status_code == 200:
            return response.json()
        return [{"doc_id": "Error", "score": 0.0, "snippet": f"Backend returned status {response.status_code}"}]
    except Exception as e:
        return [{"doc_id": "Error", "score": 0.0, "snippet": f"Could not connect to FastAPI server: {str(e)}"}]



def fetch_evaluation_metrics(model: str, limit_queries: int, top_k: int) -> dict:
    url = f"{BASE_URL}/evaluation/evaluate"


    payload = {
        "model_name": model.upper().strip(),
        "top_k": top_k
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        try:
            error_detail = response.json().get("detail", f"Status {response.status_code}")
        except:
            error_detail = f"Status {response.status_code}"
        return {"Error": f"Backend returned error: {error_detail}"}
    except Exception as e:
        return {"Error": f"Could not connect to FastAPI server: {str(e)}"}