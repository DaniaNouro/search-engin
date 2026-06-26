"""
Module Name: query_refiner.py
Purpose: Enterprise-grade Query Refinement, Formulation Assistance, and Contextual
         Retrieval pre-processing for personalized search (Implicit Feedback IR).
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import NamedTuple

import joblib

from services.retrieval.history_service import get_user_history

STORE_DIR = os.path.join("data", "vector_store")
TFIDF_VECTORIZER_PATH = os.path.join(STORE_DIR, "tfidf_vectorizer.joblib")

# Upper bound on history rows scanned per request (memory-safe, latency-bounded).
_MAX_HISTORY_SCAN = 100

# Minimum prefix length before autocomplete activates (cold-start guard).
_MIN_PREFIX_LEN = 2

# Lazy-loaded corpus vocabulary for spell-check domain adaptation only.
_vocab_words: list[str] | None = None

# ── Dynamic stopword set (English + Arabic) for intent extraction ──────────
# IR concept: Query Term Selection — remove function words that carry little
# thematic discriminative power in Vector/Lexical retrieval models.
_BASE_STOPWORDS: frozenset[str] = frozenset({
    # English function / navigational terms
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "this", "that", "these", "those", "it", "its", "they",
    "them", "their", "we", "our", "you", "your", "he", "she", "his", "her", "what",
    "which", "who", "whom", "where", "when", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "about", "into",
    "through", "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "here", "there",
    "get", "find", "buy", "make", "take", "best", "good", "new", "top", "show",
    "search", "look", "need", "want", "like", "use", "using",
    # Arabic function / navigational terms (mixed-language corpora)
    "في", "من", "على", "إلى", "الى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "هو", "هي", "هم", "نحن", "أنت", "انت", "كان",
    "كانت", "يكون", "أو", "او", "و", "لا", "ما", "لم", "لن", "قد", "كل",
    "بعض", "أي", "اي", "كيف", "متى", "أين", "اين", "لماذا", "ماذا", "هل",
})

# Punctuation stripped during token normalization; quotes preserved at word level.
_STRIP_PUNCT = "?,.:!\"'()[]{}،؛؟«»…"


class QueryFormulationResult(NamedTuple):
    """
    Structured payload returned after Implicit Query Enrichment.

    IR concepts: Query Formulation, Term Weighting, Transparent Personalization.
    The display_query preserves the user's explicit intent for UI/logging, while
    retrieval_query applies positional and frequency boosting for BM25/TF-IDF/BERT.
    """

    display_query: str
    retrieval_query: str
    context_keyword: str | None
    was_enriched: bool


def _init_vocab_cache() -> bool:
    """
    Lazily load the TF-IDF feature vocabulary for domain-adaptive spell checking.

    IR concept: Vocabulary Matching — constrains spelling corrections to in-corpus
    terms, reducing out-of-domain noise. Loads only feature names (not the full
    document-term matrix) to remain memory-safe on large vocabularies.
    """
    global _vocab_words
    if _vocab_words is not None:
        return True
    if not os.path.exists(TFIDF_VECTORIZER_PATH):
        return False
    try:
        vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        _vocab_words = list(vectorizer.get_feature_names_out())
    except Exception:
        return False
    return True


def _normalize_for_matching(text: str) -> str:
    """
    Normalize query text for consistent lexical matching across sessions.

    IR concept: Query Normalization — lowercasing and whitespace collapse improve
    Vocab Matching recall without altering the user-facing raw query string.
    """
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed


def _tokenize_query(text: str, preserve_quotes: bool = True) -> list[str]:
    """
    Tokenize a query into intent-bearing terms with punctuation protection.

    IR concept: Lexical Query Analysis — separates content terms from surface
    punctuation so stopword filtering and partial-prefix matching operate on
    clean tokens. Supports mixed English/Arabic scripts (Unicode alphabetic).
    Quoted spans are kept intact to avoid breaking exact-phrase intent.
    """
    if not text or not text.strip():
        return []

    tokens: list[str] = []
    if preserve_quotes:
        # Split on whitespace but keep quoted phrases as single tokens.
        pattern = re.compile(r'"([^"]+)"|\'([^\']+)\'|(\S+)')
        for match in pattern.finditer(text):
            quoted = match.group(1) or match.group(2)
            bare = match.group(3)
            raw = quoted if quoted is not None else bare
            if not raw:
                continue
            clean = raw.strip(_STRIP_PUNCT).lower()
            if clean:
                tokens.append(clean)
    else:
        for raw in text.split():
            clean = raw.strip(_STRIP_PUNCT).lower()
            if clean:
                tokens.append(clean)
    return tokens


def _is_valid_intent_token(token: str, min_len: int = 3) -> bool:
    """
    Determine whether a token contributes thematic intent for enrichment.

    IR concept: Query Term Selection — rejects stopwords, numeric noise, and
    ultra-short tokens that inflate DF without improving precision in lexical
    or dense retrieval models.
    """
    if not token or len(token) < min_len:
        return False
    if token.isdigit():
        return False
    if token in _BASE_STOPWORDS:
        return False
    # Accept Latin or Arabic alphabetic content (mixed-language support).
    if not any(ch.isalpha() for ch in token):
        return False
    return True


def _build_frequency_map(history: list[str]) -> dict[str, int]:
    """
    Compute implicit popularity scores from raw search_history execution logs.

    IR concept: Implicit Feedback — query frequency in the user's session log
    serves as a proxy for click-through popularity, analogous to query logs
    used in commercial Query Autocomplete rankers (e.g., Google Suggest).
    """
    freq: Counter[str] = Counter()
    for entry in history:
        normalized = _normalize_for_matching(entry)
        if normalized:
            freq[normalized] += 1
    return dict(freq)


def _compute_match_score(prefix: str, candidate: str) -> float:
    """
    Score how well a historical query matches the current typed prefix.

    IR concept: Vocab Matching + Partial Query Matching — combines:
      1) Exact prefix completion (highest confidence),
      2) Substring / infix alignment,
      3) Token-level prefix overlap (context-aware partial semantic match).

    Returns a value in [0.0, 1.0]; 0.0 means no actionable match.
    """
    if not prefix or not candidate:
        return 0.0

    p_norm = _normalize_for_matching(prefix)
    c_norm = _normalize_for_matching(candidate)

    if not p_norm or not c_norm or p_norm == c_norm:
        return 0.0

    # Tier 1 — Exact prefix completion (Google-like autocomplete).
    if c_norm.startswith(p_norm):
        return 1.0

    # Tier 2 — Infix / contains match on normalized full string.
    if p_norm in c_norm:
        return 0.85

    p_tokens = _tokenize_query(prefix)
    c_tokens = _tokenize_query(candidate)
    if not p_tokens or not c_tokens:
        return 0.0

    last_prefix = p_tokens[-1]

    # Tier 3 — Any candidate token extends the active prefix token.
    for ct in c_tokens:
        if ct.startswith(last_prefix) and ct != last_prefix:
            return 0.75

    # Tier 4 — Token overlap ratio (partial semantic relatedness).
    p_set = set(p_tokens)
    c_set = set(c_tokens)
    overlap = len(p_set & c_set)
    if overlap > 0:
        jaccard = overlap / len(p_set | c_set)
        return 0.55 + (0.25 * jaccard)

    return 0.0


def suggest_queries_from_history(
    query: str,
    user_id: str = "default_user",
    limit: int = 5,
) -> list[str]:
    """
    Context-Aware Query Autocomplete using implicit search_history signals.

    IR concepts:
      - Query Suggestion / Autocomplete: surfaces prior successful formulations
        that lexically align with the user's current prefix.
      - Implicit Feedback Ranking: orders suggestions by execution frequency
        (popularity prior) multiplied by match confidence.
      - Contextual Retrieval (personalization): reuses the user's own query log
        as a lightweight profile — no explicit relevance judgments required.

    Handles cold-start (empty history), mixed EN/AR tokens, and deduplication.
    Returns up to `limit` ranked query strings preserving original casing from
    the most recent matching log entry.
    """
    prefix = query.strip()
    if len(prefix) < _MIN_PREFIX_LEN:
        return []

    try:
        history = get_user_history(user_id, limit=_MAX_HISTORY_SCAN)
        if not history:
            return []

        frequency_map = _build_frequency_map(history)

        # Map normalized form -> (best_score, most_recent_original_text)
        ranked: dict[str, tuple[float, str]] = {}

        for idx, past_query in enumerate(history):
            if not past_query or not past_query.strip():
                continue

            normalized = _normalize_for_matching(past_query)
            match_score = _compute_match_score(prefix, past_query)
            if match_score <= 0.0:
                continue

            freq = frequency_map.get(normalized, 1)
            # Popularity prior: log-scaled frequency avoids domination by outliers.
            popularity = 1.0 + math.log1p(freq)
            final_score = match_score * popularity

            existing = ranked.get(normalized)
            # On tie, keep the first seen entry (history is DESC → most recent).
            if existing is None or final_score > existing[0]:
                ranked[normalized] = (final_score, past_query.strip())

        if not ranked:
            return []

        ordered = sorted(ranked.values(), key=lambda item: item[0], reverse=True)
        return [original for _, original in ordered[:limit]]
    except Exception as exc:
        print(f"⚠️ History autocomplete failed: {exc}")
        return []


def extract_implicit_intent_keywords(
    query: str,
    user_id: str = "default_user",
    top_n: int = 1,
) -> list[tuple[str, int]]:
    """
    Extract persistent thematic intent terms from recent implicit query history.

    IR concepts:
      - Implicit Feedback / User Profiling: aggregates content terms from past
        queries to infer latent topical interests without explicit labels.
      - Query Term Selection: filters stopwords, short tokens, digits, and terms
        already present in the current query to avoid redundant expansion.
      - Vocab Matching: retains only alphabetic EN/AR tokens suitable for
        downstream lexical (BM25, TF-IDF) and neural (BERT) encoders.

    Returns [(keyword, frequency), ...] sorted by descending frequency.
    """
    try:
        history = get_user_history(user_id, limit=_MAX_HISTORY_SCAN)
        if not history:
            return []

        current_tokens = set(_tokenize_query(query))
        intent_counter: Counter[str] = Counter()

        for past_query in history:
            for token in _tokenize_query(past_query):
                if token in current_tokens:
                    continue
                if not _is_valid_intent_token(token):
                    continue
                intent_counter[token] += 1

        if not intent_counter:
            return []

        return intent_counter.most_common(top_n)
    except Exception as exc:
        print(f"⚠️ Intent extraction failed: {exc}")
        return []


def _apply_query_reweighting(query: str, context_keyword: str) -> str:
    """
    Re-formulate the retrieval query by boosting a historical intent keyword.

    IR concept: Query Re-weighting / Term Importance — injects the profile
    keyword at the front (positional bias) and repeats it once (term-frequency
    boost) so BM25/TF-IDF saturation functions assign higher weight. BERT
    bi-encoders also benefit from leading context terms in the query embedding.

    The user's original terms remain intact; enrichment is additive only.
    """
    q = query.strip()
    kw = context_keyword.strip()
    if not q or not kw:
        return q
    # Positional + frequency boost without exposing duplication in the UI layer.
    return f"{kw} {q} {kw}"


def enrich_from_history(query: str, user_id: str = "default_user") -> str:
    """
    Implicit Query Enrichment (تثقيل الاستعلام) using search_history context.

    IR concepts:
      - Query Formulation Assistance: augments the executed query with the
        user's most frequent latent intent keyword from implicit logs.
      - Contextual Retrieval: personalizes lexical/semantic matching toward
        persistently expressed themes (long-term query profile).
      - Cold-start safety: returns the original query unchanged when history
        is empty or no discriminative intent term survives filtering.

    For structured metadata (UI transparency), use `formulate_query_with_context`.
    """
    result = formulate_query_with_context(query, user_id)
    return result.retrieval_query


def formulate_query_with_context(
    query: str,
    user_id: str = "default_user",
) -> QueryFormulationResult:
    """
    Build a dual-layer query payload for personalized retrieval execution.

    IR concepts:
      - Transparent Personalization: separates display_query (explicit user
        intent for logging/UI) from retrieval_query (re-weighted formulation).
      - Implicit Feedback Integration: selects the dominant historical intent
        keyword via frequency aggregation over filtered content terms.
      - Query Formulation pipeline stage executed immediately before indexer
        lookup across BM25, TF-IDF, BERT, and Hybrid fusion backends.

    Edge cases: empty query, empty history, mixed-language tokens, and queries
    that already contain all profile terms all resolve to a no-op enrichment.
    """
    display = query.strip()
    if not display:
        return QueryFormulationResult(
            display_query=query,
            retrieval_query=query,
            context_keyword=None,
            was_enriched=False,
        )

    intent_terms = extract_implicit_intent_keywords(display, user_id, top_n=1)
    if not intent_terms:
        return QueryFormulationResult(
            display_query=display,
            retrieval_query=display,
            context_keyword=None,
            was_enriched=False,
        )

    context_keyword, _ = intent_terms[0]
    retrieval = _apply_query_reweighting(display, context_keyword)
    return QueryFormulationResult(
        display_query=display,
        retrieval_query=retrieval,
        context_keyword=context_keyword,
        was_enriched=True,
    )


# ── Spell correction (domain-adaptive, punctuation-safe) ───────────────────

_spell = None


def _get_spell():
    """
    Lazily instantiate a corpus-aware spell checker singleton.

    IR concept: Query Refinement — orthographic normalization reduces vocabulary
    mismatch between user queries and indexed terms, improving recall in lexical
    retrieval without altering semantic intent when corrections are valid.
    """
    global _spell
    if _spell is None:
        from spellchecker import SpellChecker

        _spell = SpellChecker()
        if _init_vocab_cache() and _vocab_words:
            try:
                _spell.word_frequency.load_words(_vocab_words)
            except Exception:
                pass
    return _spell


def refine_query(query: str) -> str:
    """
    Orthographically refine a raw query while preserving punctuation and quotes.

    IR concept: Query Formulation / Lexical Refinement — performs token-level
    spelling correction constrained to in-corpus vocabulary when available.
    Short tokens, numerics, and already-valid terms pass through unchanged to
    avoid over-correction that could drift semantic intent (precision guard).
    """
    spell = _get_spell()
    if not query.strip():
        return query

    corrected: list[str] = []
    for word in query.split():
        clean = word.strip(_STRIP_PUNCT).lower()

        if not clean or clean in spell or len(clean) <= 2 or clean.isdigit():
            corrected.append(word)
            continue

        fix = spell.correction(clean)
        if fix and fix != clean:
            corrected.append(word.replace(clean, fix))
        else:
            corrected.append(word)

    return " ".join(corrected)


def suggest_expansions(query: str, top_n: int = 3) -> list[str]:
    """
    Reserved hook for corpus-level Query Expansion (pseudo-relevance feedback).

    IR concept: Query Expansion — historically augments queries with related
    index terms. Currently delegated to history-driven autocomplete and implicit
    enrichment to avoid full-vocabulary scans that risk RAM pressure on large
    TF-IDF feature spaces. Returns an empty list by design.
    """
    return []
