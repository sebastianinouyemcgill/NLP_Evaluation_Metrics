"""Load evaluation data and run the requested metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_reference_sentence_lists(reference_sentences: Any) -> List[List[str]]:
    """Accept flat ``[s1, s2, ...]`` or nested ``[[...], [...]]`` reference lists."""
    if not isinstance(reference_sentences, list) or not reference_sentences:
        raise ValueError("'reference_sentences' must be a non-empty list")

    if all(isinstance(s, str) for s in reference_sentences):
        return [reference_sentences]

    if all(isinstance(ref, list) and all(isinstance(s, str) for s in ref) for ref in reference_sentences):
        return reference_sentences

    raise ValueError("'reference_sentences' must contain strings or lists of strings")


def _records_from_sentence_bundle(item: dict[str, Any], fallback_index: int) -> List[Dict[str, Any]]:
    """Expand {source_sentences, prediction_sentences, reference_sentences} into records."""
    entry_id = item.get("id", fallback_index)
    source_sentences = item.get("source_sentences")
    prediction_sentences = item.get("prediction_sentences")
    reference_sentences = item.get("reference_sentences")

    if not isinstance(prediction_sentences, list) or not prediction_sentences:
        raise ValueError(f"Record {entry_id} must have a non-empty 'prediction_sentences' list")
    if not isinstance(source_sentences, list):
        raise ValueError(f"Record {entry_id} must have a 'source_sentences' list")
    if not all(isinstance(s, str) for s in prediction_sentences):
        raise ValueError(f"Record {entry_id} 'prediction_sentences' must contain strings")

    ref_lists = _normalize_reference_sentence_lists(reference_sentences)
    num_sentences = len(prediction_sentences)

    records: List[Dict[str, Any]] = []
    for i in range(num_sentences):
        refs = [ref_list[i] for ref_list in ref_lists if i < len(ref_list)]
        if not refs:
            raise ValueError(f"Record {entry_id} missing reference for sentence {i + 1}")

        source = source_sentences[i] if i < len(source_sentences) else None
        records.append(
            {
                "id": f"{entry_id}:{i + 1}",
                "source": source,
                "prediction": prediction_sentences[i],
                "references": refs,
            }
        )

    return records


def _record_from_sentence_row(item: dict[str, Any], fallback_index: int) -> Dict[str, Any]:
    """Normalize excerpt_to_sentences row: {page_id, sentence_id, source, prediction, reference}."""
    page_id = item.get("page_id", item.get("id", fallback_index))
    sentence_id = item.get("sentence_id", fallback_index + 1)
    prediction = item.get("prediction")
    if not isinstance(prediction, str):
        raise ValueError(f"Sentence row {page_id}:{sentence_id} must have a string 'prediction'")

    refs = item.get("references", item.get("reference"))
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list) or not refs or not all(isinstance(r, str) for r in refs):
        raise ValueError(f"Sentence row {page_id}:{sentence_id} must have a non-empty reference")

    source = item.get("source")
    if source is not None and not isinstance(source, str):
        raise ValueError(f"Sentence row {page_id}:{sentence_id} 'source' must be a string when provided")

    return {
        "id": f"{page_id}:{sentence_id}",
        "source": source,
        "prediction": prediction,
        "references": refs,
    }


def _record_from_excerpt(item: dict[str, Any], fallback_index: int) -> Dict[str, Any]:
    """Normalize excerpt-level {source, prediction, references}."""
    prediction = item.get("prediction")
    if not isinstance(prediction, str):
        raise ValueError(f"Record {fallback_index} must have a string 'prediction'")

    refs = item.get("references", item.get("reference"))
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list) or not refs or not all(isinstance(r, str) for r in refs):
        raise ValueError(f"Record {fallback_index} must have a non-empty 'references' list")

    source = item.get("source")
    if source is not None and not isinstance(source, str):
        raise ValueError(f"Record {fallback_index} 'source' must be a string when provided")

    return {
        "id": item.get("id", fallback_index),
        "source": source,
        "prediction": prediction,
        "references": refs,
    }


def _is_sentence_bundle(item: dict[str, Any]) -> bool:
    return "prediction_sentences" in item or "source_sentences" in item


def _is_sentence_row(item: dict[str, Any]) -> bool:
    return "sentence_id" in item and "page_id" in item


def load_data(path: str) -> List[Dict[str, Any]]:
    """Load excerpt, sentence-bundle, or sentence-row JSON into evaluation records."""
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = [raw]

    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of records or a single record object")

    records: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Record {i} must be a JSON object")

        if _is_sentence_bundle(item):
            records.extend(_records_from_sentence_bundle(item, i))
        elif _is_sentence_row(item):
            records.append(_record_from_sentence_row(item, i))
        else:
            records.append(_record_from_excerpt(item, i))

    if not records:
        raise ValueError("Input file contains no records")

    return records


def evaluate(
    data_path: str,
    metrics: List[str],
    bleurt_checkpoint: Optional[str] = None,
    meteor_language: str = "en",
    cometkiwi_model: str = "Unbabel/wmt22-cometkiwi-da",
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

        results["meteor"] = corpus_meteor(predictions, references, language=meteor_language)

    if "bleurt" in metrics:
        from metrics.bleurt import corpus_bleurt

        results["bleurt"] = corpus_bleurt(predictions, references, bleurt_checkpoint)

    if "cometkiwi" in metrics:
        sources = [r.get("source") for r in records]
        if not all(isinstance(s, str) for s in sources):
            raise ValueError(
                "COMETKiwi requires a string 'source' field on every record"
            )

        from metrics.cometkiwi import corpus_cometkiwi

        results["cometkiwi"] = corpus_cometkiwi(sources, predictions, model_name=cometkiwi_model)

    return results
