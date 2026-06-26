"""
Module Name: bm25_cleaner.py
Purpose: Wrapper preprocessing pipeline aligned for BM25 Probabilistic modeling.
         Ensures architectural symmetry and exact tokenization consistency with TF-IDF.
"""

import os
import sys

# Dynamic look-up optimization from the baseline lexical cleaning ecosystem
from services.preprocessing.tfidf_cleaner import tfidf_custom_tokenizer

def bm25_custom_tokenizer(text: str) -> list[str]:
    """
    Unified tokenizer wrapper for the BM25 pipeline to guarantee strict lexical
    consistency and systemic cross-model alignment during performance evaluation.

    IR Core Operation:
      Systemic Symmetry: Forcing the exact same text normalization, stopword filtering,
      and Porter Stemming across both TF-IDF and BM25 models eliminates vocabulary
      mismatch noise, making downstream evaluation metrics (MAP, NDCG, Precision@K)
      strictly dependent on the scoring algorithms themselves.

    Args:
        text (str): Raw or base-cleaned textual string document or user query.

    Returns:
        list[str]: Collection of conflated core linguistic stems.

    Example:
        >>> input_query = "The probabilistic retrieval architectures are analyzing inputs!"
        >>> bm25_custom_tokenizer(input_query)
        Output: ['probabilist', 'retriev', 'architectur', 'analyz', 'input']
    """
    return tfidf_custom_tokenizer(text)