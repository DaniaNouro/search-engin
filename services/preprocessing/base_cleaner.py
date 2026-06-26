"""
Module Name: text_normalizer.py
Library Used: Standard Python Libraries ('re' for Regular Expressions, 'string' for core text utilities).
Purpose: Performs conservative text normalization. It converts characters to lowercase, removes hidden control characters,
         normalizes whitespaces, and cleans punctuation carefully to preserve meaningful compound words like 'state-of-the-art' or 'COVID-19'.
"""

import re

def lowercase_text(text: str) -> str:
    """
    Transforms all characters in the input string to lowercase.
    Why: To ensure case-insensitivity across lexical models (BM25) and semantic models (BERT).

    Example:
        Input: "The BERT-base Model"
        Output: "the bert-base model"
    """
    return text.lower()


def remove_control_characters(text: str) -> str:
    """
    Removes hidden ASCII control characters and non-printable bytes (e.g., \x00-\x1F, \x7F-\x9F).
    Why: These characters cause tokenization glitches, database encoding errors, and terminal crashes.

    Example:
        Input: "Document\x9d Text\x00"
        Output: "Document  Text "
    """
    return re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)


def remove_basic_punctuation_conservative(text: str) -> str:
    """
    Cleans standard punctuation marks but conservatively preserves internal hyphens and underscores.
    Replaces unneeded symbols with spaces instead of deleting them to prevent word-clumping.

    Why: Deleting hyphens turns 'state-of-the-art' into 'stateoftheart', which destroys the token's semantic meaning.

    Example:
        Input: "Hello, World! This is a state-of-the-art system (v2.0)."
        Output: "Hello  World  This is a state-of-the-art system  v2 0  "
    """
    # [^\w\s-] matches any character that is NOT a word character (letters/digits), whitespace, or a hyphen.
    return re.sub(r"[^\w\s-]", " ", text)


def normalize_whitespaces(text: str) -> str:
    """
    Removes redundant spaces, tabs, and newlines, compressing them into a single space.
    Why: Ensures clean token boundaries and efficient string storage.

    Example:
        Input: "regular   prose    and   text "
        Output: "regular prose and text"
    """
    return " ".join(text.split())


def light_cleaning_pipeline(raw_text: str) -> str:
    """
    Executes the optimized light preprocessing chain. Highly flexible and safe for Hybrid Search.

    Example:
        Input: "⚠️ CRITICAL: COVID-19 deployment failed!!! \n Check Log-file #4\x9d "
        Output: "critical covid-19 deployment failed check log-file 4"
    """
    # Step 1: Lowercase
    cleaned = lowercase_text(raw_text)

    # Step 2: Clear hidden bytes
    cleaned = remove_control_characters(cleaned)

    # Step 3: Conservative punctuation removal
    cleaned = remove_basic_punctuation_conservative(cleaned)

    # Step 4: Squash extra spaces
    cleaned = normalize_whitespaces(cleaned)

    return cleaned

# --- Quick Local Test ---
if __name__ == "__main__":
    test_sentence = "  BERT-base setup: Processing COVID-19 dataset with state-of-the-art tools!!!\x9d "
    print("Original:", test_sentence)
    print("Cleaned :", light_cleaning_pipeline(test_sentence))