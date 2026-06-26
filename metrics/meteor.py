"""
Corpus-level METEOR via NLTK.

Uses NLTK's meteor_score, which implements the METEOR 1.5 algorithm with
English WordNet synonyms and an English stemmer. It is English-only: use it
when candidate and reference text are in English. For other target languages,
METEOR is not computed by this module.
"""

from __future__ import annotations

from typing import List

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score


def _ensure_nltk_data() -> None:
    import nltk

    for package in ("wordnet", "punkt", "punkt_tab", "omw-1.4"):
        try:
            if package.startswith("punkt"):
                nltk.data.find(f"tokenizers/{package}")
            else:
                nltk.data.find(f"corpora/{package}")
        except LookupError:
            nltk.download(package, quiet=True)


def corpus_meteor(predictions: List[str], references: List[List[str]]) -> float:
    """Return mean sentence-level METEOR (scores are typically between 0 and 1)."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    _ensure_nltk_data()

    scores = []
    for pred, refs in zip(predictions, references):
        pred_tokens = word_tokenize(pred)
        ref_tokens = [word_tokenize(r) for r in refs]
        scores.append(meteor_score(ref_tokens, pred_tokens))

    return sum(scores) / len(scores)
