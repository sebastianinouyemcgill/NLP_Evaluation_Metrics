"""Load evaluation data and run the requested metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_data(path: str) -> List[Dict[str, Any]]:
    """Load records from a JSON file.

    Expected format:
        [{"id": 1, "prediction": "...", "references": ["...", "..."]}, ...]
    """
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of records")

    records = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Record {i} must be a JSON object")

        prediction = item.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError(f"Record {i} must have a string 'prediction'")

        refs = item.get("references", item.get("reference"))
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list) or not refs or not all(isinstance(r, str) for r in refs):
            raise ValueError(f"Record {i} must have a non-empty 'references' list")

        records.append(
            {
                "id": item.get("id", i),
                "prediction": prediction,
                "references": refs,
            }
        )

    if not records:
        raise ValueError("Input file contains no records")

    return records


def evaluate(
    data_path: str,
    metrics: List[str],
    bleurt_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Run metrics and return {"num_examples": N, "<metric>": score, ...}."""
    records = load_data(data_path)
    predictions = [r["prediction"] for r in records]
    references = [r["references"] for r in records]

    results: Dict[str, Any] = {"num_examples": len(records)}

    if "bleu" in metrics:
        from metrics.bleu import corpus_bleu

        results["bleu"] = corpus_bleu(predictions, references)

    if "meteor" in metrics:
        from metrics.meteor import corpus_meteor

        results["meteor"] = corpus_meteor(predictions, references)

    if "bleurt" in metrics:
        from metrics.bleurt import corpus_bleurt

        results["bleurt"] = corpus_bleurt(predictions, references, bleurt_checkpoint)

    return results
