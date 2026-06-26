"""
Module Name: tfidf_cleaner.py
Purpose: Comprehensive deep preprocessing pipeline customized for Lexical TF-IDF modeling.
Features: Integrates foundational light cleaning, advanced NLTK word tokenization,
          strict English stopword filtering, and grammatical Porter Stemming.
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Absolute lookup for the baseline unified text normalization stage
from services.preprocessing.base_cleaner import light_cleaning_pipeline

# Ensure necessary NLTK resource packages are cached safely to guarantee offline runtime stability
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


def tfidf_custom_tokenizer(text: str) -> list[str]:
    """
    Executes a multi-stage advanced text preprocessing pipeline tailored for spatial Vector
    Space Models (VSM) and Lexical TF-IDF Indexing.

    IR Core Operations:
      1. Text Normalization: Invokes the base pipeline to strip control characters and lower-case tokens,
         minimizing surface vocabulary variance.
      2. Lexical Analysis (Tokenization): Uses NLTK's Treebank Word Tokenizer to isolate alphanumeric
         content and keep compound domain markers (like dashes) structurally unified.
      3. Query Term Selection (Stopword Filtering): Strips high-frequency English function words
         (e.g., 'the', 'is') that inflate Document Frequency (DF) without carrying discriminative theme power.
      4. Conflation (Porter Stemming): Maps morphologically related variants to their root linguistic stem,
         reducing vocabulary dimensionality and significantly improving Recall during term matching.

    Args:
        text (str): Raw unstructured textual document string or search query payload.

    Returns:
        list[str]: Array of normalized, stemmed content tokens optimized for spatial VSM indexing.

    Example:
        >>> input_text = "The software engineers are compiling data and optimizing queries smoothly!"
        >>> tfidf_custom_tokenizer(input_text)
        >>> # Step 1 & 2 (Tokenize): ['the', 'software', 'engineers', 'are', 'compiling', 'data', 'and', 'optimizing', 'queries', 'smoothly']
        >>> # Step 3 (Stopwords Filtered): ['software', 'engineers', 'compiling', 'data', 'optimizing', 'queries', 'smoothly']
        >>> # Step 4 (Porter Stemming Applied):
        Output: ['softwar', 'engin', 'compil', 'data', 'optim', 'queri', 'smoothli']
    """
    if not text or not isinstance(text, str):
        return []

    # ── Step 1: Text Normalization ──
    # Cleans structural noise, controls layout anomalies, and standardizes casing.
    normalized_text = light_cleaning_pipeline(text)

    # ── Step 2: Linguistic Tokenization ──
    # Separates text into precise atomic word components using punctuation boundary awareness.
    tokens = word_tokenize(normalized_text)

    # ── Step 3 & 4: Stopword Trimming & Morphological Conflation (Stemming) ──
    # Isolates vocabulary terms using dynamic lookups and strips suffixes algorithmically via Porter rules.
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    cleaned_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and token.strip()
    ]

    return cleaned_tokens