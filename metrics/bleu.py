"""Corpus-level BLEU via SacreBLEU (WMT-standard tokenization)."""

from __future__ import annotations

from typing import List

import sacrebleu


def corpus_bleu(predictions: List[str], references: List[List[str]]) -> float:
    """Return corpus BLEU on a 0–100 scale (SacreBLEU convention)."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    max_refs = max(len(r) for r in references)
    padded = [r + [r[0]] * (max_refs - len(r)) for r in references]
    transposed = [list(col) for col in zip(*padded)]

    return sacrebleu.corpus_bleu(predictions, transposed).score
