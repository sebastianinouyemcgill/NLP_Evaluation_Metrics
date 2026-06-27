"""Corpus-level BLEURT (TensorFlow). Requires a downloaded checkpoint directory."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def corpus_bleurt(
    predictions: List[str],
    references: List[List[str]],
    checkpoint_path: Optional[str],
) -> float:
    """Return mean sentence-level BLEURT (multi-reference: max score per example)."""
    if not checkpoint_path:
        raise ValueError(
            "BLEURT requires a checkpoint directory. Download a checkpoint "
            "from https://github.com/google-research/bleurt (e.g. BLEURT-20-D12) "
            "and pass --bleurt-checkpoint /path/to/checkpoint"
        )

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"BLEURT checkpoint directory not found: {checkpoint}")

    try:
        from bleurt import score
    except ImportError as e:
        raise ImportError(
            "BLEURT is not installed. Install dependencies with "
            "pip install -r requirements.txt"
        ) from e

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    scorer = score.BleurtScorer(str(checkpoint))
    sentence_scores = []
    for pred, refs in zip(predictions, references):
        pair_scores = scorer.score(references=refs, candidates=[pred] * len(refs))
        sentence_scores.append(max(pair_scores))

    return sum(sentence_scores) / len(sentence_scores)
