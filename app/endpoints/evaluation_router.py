"""
Module Name: evaluation_router.py
Purpose: FastAPI router defining HTTP endpoints for system batch evaluation and metrics.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from services.evaluation.evaluation_service import evaluate_model

router = APIRouter(
    prefix="/evaluation",
    tags=["System Evaluation Engine"]
)



class EvaluationRequest(BaseModel):
    model_name: str = Field(..., example="TF-IDF")
    top_k: int = Field(10, ge=1, le=50)


@router.post("/evaluate")
def api_evaluate_model(payload: EvaluationRequest):
    """
    Triggers batch system evaluation on ground-truth data to compute MAP, Precision, and Recall.
    """
    metrics = evaluate_model(
        model_name=payload.model_name,
        # limit_queries=payload.limit_queries,
        top_k=payload.top_k
    )

    if "Error" in metrics:
        raise HTTPException(status_code=500, detail=metrics["Error"])

    return metrics